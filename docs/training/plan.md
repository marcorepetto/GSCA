# Plan de Implementación: Función de Pérdida y Entrenamiento (Circle Loss con Self-paced Weighting)

Este documento detalla la planificación y diseño técnico para la implementación del módulo de **Función de Pérdida y Entrenamiento** dentro del proyecto GSCA (Geo-Structural Cross-Attention). Este módulo es responsable de guiar el aprendizaje de representaciones conjuntas para el matching 2D-3D en afloramientos geológicos, mitigando las limitaciones de funciones de pérdida clásicas ante la redundancia estructural fractal.

---

## 1. Propósito y Contexto del Módulo

En afloramientos geológicos, las estructuras físicas como fracturas, estratos sedimentarios y planos de falla exhiben patrones cíclicos y de auto-similitud a distintas escalas (carácter fractal). Esta redundancia estructural provoca que descriptores en zonas geológicamente diferentes pero visualmente similares tiendan a confundirse, lo que representa un reto crítico para los modelos de alineamiento tradicionales.

### El Rol de Circle Loss con Self-paced Weighting
El módulo de pérdida de GSCA implementa **Circle Loss con Self-paced Weighting** para abordar estas limitaciones:
1. **Frontera de Decisión Circular y Flexible**: A diferencia de *Triplet Loss* u otras pérdidas de margen rígido (que penalizan con la misma fuerza a pares fáciles y difíciles dentro del margen), Circle Loss pondera de forma independiente la dificultad de cada par.
2. **Currículo de Aprendizaje Dinámico (Self-paced)**:
   - **Pares Consolidados (Fáciles)**: Si un par positivo ya posee una alta similitud ($s_p \approx O_p$), su factor de ponderación $\alpha_p$ se reduce a cero, deteniendo la optimización de ese par y protegiendo los descriptores ya alineados.
   - **Pares Ambiguos (Difíciles)**: Si un par negativo se encuentra muy cercano en el espacio latente (alta similitud incorrecta $s_n$), su factor $\alpha_n$ se incrementa drásticamente, amplificando el gradiente para repeler sus descriptores y desambiguar la estructura.

### Posición en la Arquitectura
Este módulo actúa en la etapa final de la propagación hacia adelante (*forward pass*) durante el entrenamiento. Recibe los descriptores latentes de la **Rama Visual (2D)** y la **Rama Geométrica (3D)** y, basándose en la correspondencia del terreno (*Ground Truth* de proyección), calcula la pérdida escalar utilizada para actualizar mediante retropropagación (*backpropagation*):
- Los **Visual Adapters** y el decodificador **FPN** de la Rama Visual 2D.
- El extractor de características **DGCNN** (capas EdgeConv) de la Rama Geométrica 3D.
- El backbone principal DINOv2 permanece **congelado** para evitar el olvido catastrófico de características visuales generales.

---

## 2. Especificación Estricta de Interfaces

El módulo opera a nivel de lote (*batch*) procesando las características latentes y la máscara de correspondencias verdaderas.

### Tensores de Entrada

| Tensor | Forma (*Shape*) | Tipo de Datos | Rango de Valores | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `feat_2d` | `[B, N_2D, C]` | `Float32` | `[-1.0, 1.0]` (Normalizado $L_2$) | Descriptores visuales densos proyectados al espacio latente común $C$ para un lote de tamaño $B$. $N_{2D}$ representa el número de píxeles/parches seleccionados. |
| `feat_3d` | `[B, N_3D, C]` | `Float32` | `[-1.0, 1.0]` (Normalizado $L_2$) | Descriptores geométricos de la nube de puntos proyectados al espacio latente común $C$. $N_{3D}$ es la cantidad de puntos 3D procesados. |
| `gt_mask` | `[B, N_2D, N_3D]` | `Bool` | `{False, True}` | Máscara de correspondencia real del terreno. `True` si el píxel $i$ en 2D corresponde al punto $j$ en 3D (determinado mediante la pose de cámara exacta $\mathbf{T}_{gt}$ y calibración $\mathbf{K}_{cam}$). |

### Parámetros de Configuración (Hiperparámetros)

- `gamma` ($\gamma$): Factor de escala continuo. Modula la nitidez de la distribución de probabilidad suave. Rango recomendado: `[64.0, 128.0]` (por defecto `80.0`).
- `margin` ($\Delta$): Margen de holgura para la separación. Controla el límite de decisión. Rango recomendado: `[0.2, 0.3]` (por defecto `0.25`).
  - Determina los objetivos de convergencia óptimos:
    - $O_p = 1.0 + \Delta$
    - $O_n = -\Delta$
  - Determina los márgenes de holgura aplicados:
    - $\Delta_p = 1.0 - \Delta$
    - $\Delta_n = \Delta$

### Tensores de Salida

| Tensor | Forma (*Shape*) | Tipo de Datos | Rango de Valores | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `loss` | `[]` (Escalar) | `Float32` | `[0.0, +inf)` | Valor acumulado de la pérdida para el lote. Requiere cálculo de gradiente (`requires_grad=True` durante entrenamiento). |

---

## 3. Flujo Lógico y Diseño Algorítmico

### 3.1 Proceso de Cálculo de Pérdida (Pseudocódigo Conceptual)

El flujo del algoritmo se describe a continuación de manera conceptual, enfocándose en la estabilidad numérica y el procesamiento por lotes:

```text
ALGORITMO CalcularCircleLoss
    ENTRADAS: 
        feat_2d: Tensor [B, N_2D, C]
        feat_3d: Tensor [B, N_3D, C]
        gt_mask: Tensor de Booleano [B, N_2D, N_3D]
        gamma: Real
        margin: Real
    
    SALIDAS:
        loss: Escalar Real

    Definir O_p = 1.0 + margin
    Definir O_n = -margin
    Definir Delta_p = 1.0 - margin
    Definir Delta_n = margin
    
    Inicializar lista_perdidas_batch = []
    
    PARA CADA elemento b DESDE 0 HASTA B-1 HACER:
        f2d_b = feat_2d[b]
        f3d_b = feat_3d[b]
        mask_b = gt_mask[b]
        
        sim_matrix = ProductoPunto(f2d_b, Transponer(f3d_b))
        
        pares_positivos = sim_matrix[mask_b]
        pares_negativos = sim_matrix[NO mask_b]
        
        SI Longitud(pares_positivos) == 0 O Longitud(pares_negativos) == 0 ENTONCES:
            Añadir 0.0 a lista_perdidas_batch
            CONTINUAR
            
        alpha_p = Rectificar(O_p - DetenerGradiente(pares_positivos))
        alpha_n = Rectificar(DetenerGradiente(pares_negativos) - O_n)
        
        logits_p = -gamma * alpha_p * (pares_positivos - Delta_p)
        logits_n = gamma * alpha_n * (pares_negativos - Delta_n)
        
        loss_p = LogSumExp(logits_p)
        loss_n = LogSumExp(logits_n)
        
        loss_b = Log1p(Exp(loss_p + loss_n))
        
        Añadir loss_b a lista_perdidas_batch
        
    FIN PARA
    
    loss_promedio = Promediar(lista_perdidas_batch)
    
    RETORNAR loss_promedio
FIN ALGORITMO
```

### 3.2 Estrategia de Estabilidad Numérica
El factor de escala $\gamma$ provoca que las funciones exponenciales se saturen rápidamente, llevando a desbordamientos numéricos. La implementación debe usar estrictamente:
- La función integrada de reducción **Log-Sum-Exp** de PyTorch (`torch.logsumexp`) la cual realiza el truco de la resta del máximo de forma transparente.
- La función **Log1p** ($\log(1+x)$) para computar de forma precisa el término logarítmico exterior cuando el argumento exponencial es cercano a cero.

### 3.3 Flujo de Optimización y Retropropagación
1. **Paso de Gradiente**: Se calcula el gradiente del escalar `loss` respecto a los descriptores `feat_2d` y `feat_3d`.
2. **Propagación en la Rama Visual (2D)**:
   - Los gradientes fluyen desde la salida de la FPN hacia los **Visual Adapters** paralelos a las capas MLP de DINOv2.
   - El grafo de cómputo se detiene en los bloques originales de DINOv2 (los cuales permanecen congelados).
3. **Propagación en la Rama Geométrica (3D)**:
   - Los gradientes fluyen hacia los descriptores del extractor de puntos 3D a través de los bloques convolucionales dinámicos de grafos (`DynamicEdgeConv`), actualizando los pesos de los perceptrones locales $h_{\Theta}$.

### 3.4 Configuración del Planificador de Entrenamiento
- **Optimizador**: AdamW con tasa de aprendizaje base $\eta = 10^{-4}$ y decaimiento de peso (*weight decay*) de $10^{-4}$.
- **Warm-up**: Incremento lineal de la tasa de aprendizaje durante las primeras 5 épocas desde $10^{-6}$ hasta la tasa base $\eta$.
- **Scheduler**: Recocido Coseno (*Cosine Annealing*) para disminuir gradualmente la tasa de aprendizaje hasta un mínimo de $10^{-6}$.

---

## 4. Dependencias e Integración

1. **PyTorch (core)**: Operaciones matemáticas fundamentales sobre tensores.
2. **Rama Visual 2D (FPN y Adapters)**: Provee los descriptores visuales densos en la forma `[B, N_2D, C]`.
3. **Rama Geométrica 3D (DGCNN)**: Provee los descriptores geométricos en la forma `[B, N_3D, C]`.
4. **Módulo de Proyección y Geometría (GSCA)**: Suministra la máscara booleana `gt_mask`.

---

## 5. Estrategia y Diseño de Pruebas Unitarias

1. **Prueba de Alineamiento Perfecto (Caso Óptimo)**: Descriptores idénticos para correspondencias reales. Se verifica que $\alpha_p \to 0.0$ y la pérdida tienda a su mínimo teórico.
2. **Prueba de Desalineamiento Total (Caso de Máxima Pérdida)**: Similitud negativa para positivos y máxima para negativos. Se evalúa que la pérdida alcance su cota máxima y el gradiente sea alto.
3. **Prueba de Estabilidad Numérica**: Evaluación con $\gamma = 1000.0$ comprobando que no se generen `NaN` o `inf`.
4. **Prueba de Casos de Esquina de Máscara (Lote Sin Correspondencias)**: Verificación de que el código maneje de manera segura cuando no hay positivos o negativos, retornando `0.0` y manteniendo `requires_grad=True`.
5. **Prueba de Flujo y Propagación de Gradientes**: Ejecución de `backward()` para confirmar que los gradientes llegan a los descriptores de entrada.
