# Lista de Tareas: Función de Pérdida y Entrenamiento (Circle Loss con Self-paced Weighting)

Este documento define el desglose de tareas técnicas atómicas y secuenciales necesarias para implementar la función de pérdida y el pipeline de entrenamiento del proyecto GSCA, a partir del plan de implementación estructurado en `plan.md`.

---

## Índice de Tareas

1. [T-LOSS-01: Implementación del módulo de pérdida CircleLossWithSelfPacedWeighting](#t-loss-01-implementación-del-módulo-de-pérdida-circlelosswithselfpacedweighting)
2. [T-LOSS-02: Diseño y ejecución de pruebas unitarias para CircleLossWithSelfPacedWeighting](#t-loss-02-diseño-y-ejecución-de-pruebas-unitarias-para-circlelosswithselfpacedweighting)
3. [T-TRAIN-01: Configuración del optimizador AdamW y planificador (Scheduler)](#t-train-01-configuración-del-optimizador-adamw-y-planificador-scheduler)
4. [T-TRAIN-02: Diseño y ejecución de pruebas unitarias para el optimizador y planificador](#t-train-02-diseño-y-ejecución-de-pruebas-unitarias-para-el-optimizador-y-planificador)
5. [T-TRAIN-03: Implementación del paso de entrenamiento (train_step)](#t-train-03-implementación-del-paso-de-entrenamiento-train_step)
6. [T-TRAIN-04: Diseño y ejecución de pruebas unitarias para el paso de entrenamiento](#t-train-04-diseño-y-ejecución-de-pruebas-unitarias-para-el-paso-de-entrenamiento)

---

## Tareas Detalladas

### T-LOSS-01: Implementación del módulo de pérdida CircleLossWithSelfPacedWeighting

- **ID de la Tarea**: `T-LOSS-01`
- **Título**: Implementación del módulo de pérdida `CircleLossWithSelfPacedWeighting`
- **Descripción**: 
  Desarrollar una clase en PyTorch que herede de `torch.nn.Module` para computar la Circle Loss con Self-paced Weighting sobre pares de descriptores densos 2D-3D. La clase procesa la similitud de pares positivos y negativos a partir de una máscara de correspondencia real, calcula dinámicamente las ponderaciones $\alpha_p$ y $\alpha_n$ con detención de flujo de gradiente, y aplica un esquema de estabilidad numérica estricto para evitar desbordamientos aritméticos por el escalamiento con el hiperparámetro $\gamma$.
- **Requisitos Técnicos Duros**:
  - **Firma de la clase**: `class CircleLossWithSelfPacedWeighting(torch.nn.Module):`
  - **Constructor**:
    - Firma: `def __init__(self, gamma: float = 80.0, margin: float = 0.25)`
    - Validar que `gamma > 0` y `0.0 < margin < 1.0`. Almacenar los valores de holgura teóricos:
      - $O_p = 1.0 + margin$
      - $O_n = -margin$
      - $\Delta_p = 1.0 - margin$
      - $\Delta_n = margin$
  - **Método Forward**:
    - Firma: `def forward(self, feat_2d: torch.Tensor, feat_3d: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor`
    - **Tipos y dimensiones esperadas de entrada**:
      - `feat_2d`: `torch.Tensor` de tipo `torch.float32`, forma `[B, N_2D, C]`.
      - `feat_3d`: `torch.Tensor` de tipo `torch.float32`, forma `[B, N_3D, C]`.
      - `gt_mask`: `torch.Tensor` de tipo `torch.bool`, forma `[B, N_2D, N_3D]`.
    - **Tipo de retorno**: `torch.Tensor` escalar de tipo `torch.float32`, forma `[]` (cero dimensiones).
  - **Cálculo Matemático por Lote**:
    - Para cada elemento $b$ en el lote de tamaño $B$:
      1. Calcular la matriz de similitud de producto punto entre descriptores: `sim_matrix = torch.matmul(feat_2d[b], feat_3d[b].t())` (asumiendo descriptores pre-normalizados $L_2$ en la dimensión $C$).
      2. Separar similitudes de pares positivos $s_p = sim\_matrix[gt\_mask[b]]$ y negativos $s_n = sim\_matrix[\sim gt\_mask[b]]$.
      3. Control de caso límite: Si la cantidad de elementos en $s_p$ o $s_n$ es cero, la pérdida para este lote `loss_b` debe inicializarse en `0.0` (como un tensor en el mismo dispositivo de las entradas con `requires_grad=True`), y continuar al siguiente elemento del lote.
      4. Calcular factores de peso dinámicos separando el gradiente:
         - $\alpha_p = \text{relu}(O_p - s_p\text{.detach()})$
         - $\alpha_n = \text{relu}(s_n\text{.detach()} - O_n)$
      5. Calcular los logits escalados:
         - $logits_p = -\gamma \cdot \alpha_p \cdot (s_p - \Delta_p)$
         - $logits_n = \gamma \cdot \alpha_n \cdot (s_n - \Delta_n)$
      6. Obtener la pérdida del lote de forma estable:
         - $loss\_p = \text{torch.logsumexp}(logits_p, \text{dim}=0)$
         - $loss\_n = \text{torch.logsumexp}(logits_n, \text{dim}=0)$
         - $loss\_b = \text{torch.log1p}(\text{torch.exp}(loss\_p + loss\_n))$
    - La pérdida final de salida es el promedio de todas las pérdidas de lote individuales: $\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} loss\_b$.
- **Criterios de Aceptación**:
  - La función retorna un tensor escalar (`shape=[]`) que conserva el historial de computación para la retropropagación (`requires_grad=True`).
  - La pérdida no debe producir valores nulos (`NaN`) o infinitos (`inf`) bajo condiciones de entrada correctas de rango $[-1.0, 1.0]$ para los descriptores.
  - La lógica debe estar optimizada en PyTorch y no contener bucles manuales que ralenticen drásticamente la GPU (se permite el bucle sobre el lote $B$ dado que el número de pares válidos varía dinámicamente por elemento del lote, pero las operaciones internas de similitud y reducción deben resolverse de manera completamente vectorizada).

---

### T-LOSS-02: Diseño y ejecución de pruebas unitarias para CircleLossWithSelfPacedWeighting

- **ID de la Tarea**: `T-LOSS-02`
- **Título**: Diseño y ejecución de pruebas unitarias para `CircleLossWithSelfPacedWeighting`
- **Descripción**: 
  Implementar pruebas automatizadas unitarias exhaustivas con `pytest` para garantizar el correcto funcionamiento matemático del módulo de pérdida, el comportamiento de sus gradientes y su resistencia ante casos límite e inestabilidades numéricas.
- **Requisitos Técnicos Duros**:
  - **Ubicación del archivo**: `tests/training/test_circle_loss.py`
  - **Casos de prueba obligatorios**:
    1. **Alineamiento Perfecto (`test_perfect_alignment`)**:
       - Generar descriptores sintéticos idénticos para pares correspondientes (similitud $s_p = 1.0$) y ortogonales/opuestos para los negativos (similitud $s_n \le 0.0$).
       - Verificar que los factores de ponderación $\alpha_p$ calculados tiendan a $0.0$ y que la pérdida resultante sea mínima e inferior a $0.05$.
    2. **Desalineamiento Extremo (`test_maximal_misalignment`)**:
       - Generar descriptores sintéticos donde $s_p = -1.0$ (máximo error positivo) y $s_n = 1.0$ (máximo error negativo).
       - Verificar que la pérdida final sea alta y que los gradientes de retropropagación sean de magnitud significativa.
    3. **Estabilidad Numérica con Escala Grande (`test_numerical_stability`)**:
       - Instanciar la pérdida con $\gamma = 1000.0$. Pasar tensores con valores en los límites de rango $[-1.0, 1.0]$.
       - Comprobar que no se generen valores `NaN` o `inf` y que la pérdida retorne un valor finito.
    4. **Máscaras Vacías y Llenas (`test_corner_cases_mask`)**:
       - Probar con una máscara de correspondencias vacía (`all False`) y otra llena (`all True`).
       - Verificar que la pérdida sea calculada como `0.0` y que permita ejecutar el método `.backward()` sin fallos.
    5. **Propagación del Gradiente (`test_gradient_propagation`)**:
       - Asegurar que `feat_2d` y `feat_3d` tengan activado `requires_grad=True`.
       - Calcular la pérdida, ejecutar `.backward()` y verificar que los gradientes de entrada existan, tengan formas idénticas a las entradas y no contengan valores indefinidos.
- **Criterios de Aceptación**:
  - La suite de pruebas debe ejecutarse con `pytest tests/training/test_circle_loss.py` y pasar al 100%.
  - Se debe obtener una cobertura de código de al menos el 98% en el archivo de implementación de la pérdida.

---

### T-TRAIN-01: Configuración del optimizador AdamW y planificador (Scheduler)

- **ID de la Tarea**: `T-TRAIN-01`
- **Título**: Configuración del optimizador AdamW y planificador (Scheduler) con Warm-up y Cosine Annealing
- **Descripción**: 
  Escribir la función de configuración del optimizador y planificador de tasa de aprendizaje. Esta función debe realizar la separación de los parámetros entrenables y congelados en el modelo GSCA unificado, y estructurar el decaimiento de la tasa de aprendizaje cumpliendo con el warm-up lineal y la función coseno.
- **Requisitos Técnicos Duros**:
  - **Firma de la función**: 
    `def configure_optimizers(model: torch.nn.Module, epochs: int, steps_per_epoch: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:`
  - **Congelamiento de Parámetros**:
    - Identificar las capas correspondientes a DINOv2 y configurar explícitamente `requires_grad = False` para todos sus parámetros.
    - Asegurar que los parámetros de los Visual Adapters, decodificador FPN y el extractor DGCNN (capas EdgeConv) mantengan `requires_grad = True`.
    - Pasar únicamente los parámetros activos (`requires_grad == True`) al optimizador.
  - **Configuración del Optimizador**:
    - Instanciar `torch.optim.AdamW`.
    - Tasa de aprendizaje base ($\eta$): $1.0 \times 10^{-4}$.
    - Decaimiento de pesos (weight decay): $1.0 \times 10^{-4}$.
  - **Configuración del Scheduler**:
    - El scheduler debe calcular la tasa de aprendizaje por paso global de optimización (iteración), no por época, para garantizar transiciones suaves.
    - Pasos de Warm-up ($T_{warmup}$): $5 \times steps\_per\_epoch$.
    - Pasos Totales ($T_{max}$): $epochs \times steps\_per\_epoch$.
    - **Fórmula de Tasa de Aprendizaje ($\eta_{step}$)**:
      - Si $step < T_{warmup}$:
        $$\eta_{step} = \eta_{min} + \left(\frac{step}{T_{warmup}}\right) \cdot (\eta_{base} - \eta_{min})$$
        Donde $\eta_{base} = 10^{-4}$ y $\eta_{min} = 10^{-6}$.
      - Si $step \ge T_{warmup}$:
        $$\eta_{step} = \eta_{min} + \frac{1}{2}(\eta_{base} - \eta_{min})\left(1 + \cos\left(\pi \frac{step - T_{warmup}}{T_{max} - T_{warmup}}\right)\right)$$
- **Criterios de Aceptación**:
  - La función retorna una tupla válida `(optimizer, scheduler)` de PyTorch lista para integrarse en un bucle de entrenamiento estándar.
  - No se incluye ningún parámetro perteneciente al backbone congelado en la lista del optimizador (se debe validar mediante validaciones de tipo `assert` en el código).

---

### T-TRAIN-02: Diseño y ejecución de pruebas unitarias para el optimizador y planificador

- **ID de la Tarea**: `T-TRAIN-02`
- **Título**: Diseño y ejecución de pruebas unitarias para el optimizador y planificador
- **Descripción**: 
  Implementar pruebas en `pytest` para verificar la congelación selectiva de capas y comprobar que la curva de tasa de aprendizaje generada por el planificador siga estrictamente las especificaciones de warm-up y decaimiento coseno.
- **Requisitos Técnicos Duros**:
  - **Ubicación del archivo**: `tests/training/test_optimization.py`
  - **Casos de prueba obligatorios**:
    1. **Congelación Selectiva (`test_parameter_freezing`)**:
       - Instanciar un modelo GSCA mock que simule tener parámetros en `backbone.dinov2`, `visual_adapters`, `fpn_decoder` y `dgcnn`.
       - Ejecutar la configuración de optimizadores.
       - Verificar que los parámetros del backbone tengan `requires_grad == False`.
       - Verificar que el optimizador tenga en su lista de parámetros solo aquellos con `requires_grad == True`.
    2. **Perfil del Scheduler (`test_learning_rate_schedule`)**:
       - Simular un entorno de entrenamiento de 10 épocas con 100 pasos por época.
       - Iterar paso a paso invocando `scheduler.step()` y recuperando la tasa de aprendizaje mediante `scheduler.get_last_lr()[0]`.
       - Verificar que:
         - En el paso 0, la tasa sea aproximadamente $10^{-6}$.
         - En el paso 500 (final de la época 5), la tasa sea exactamente $10^{-4}$.
         - En el paso 1000 (final del entrenamiento), la tasa sea exactamente $10^{-6}$.
         - Las tasas intermedias entre el paso 500 y 1000 sigan una curva descendente monótona no lineal y simétrica de tipo coseno.
- **Criterios de Aceptación**:
  - Las pruebas unitarias deben ejecutarse de forma rápida sin dependencias de hardware GPU ni archivos preentrenados de gran escala.
  - Todas las pruebas deben pasar exitosamente.

---

### T-TRAIN-03: Implementación del paso de entrenamiento (train_step)

- **ID de la Tarea**: `T-TRAIN-03`
- **Título**: Implementación del paso de entrenamiento unitario
- **Descripción**: 
  Desarrollar la lógica del paso de entrenamiento que encapsula el ciclo completo de procesamiento para una iteración (propagación hacia adelante, cálculo de la Circle Loss, retropropagación de gradientes, paso del optimizador y del scheduler, y recolección de estadísticas).
- **Requisitos Técnicos Duros**:
  - **Firma de la función**:
    `def train_step(model: torch.nn.Module, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, loss_fn: torch.nn.Module, device: torch.device) -> Dict[str, float]:`
  - **Parámetros de Entrada**:
    - `model`: Modelo de red neuronal unificado GSCA.
    - `batch`: Diccionario que contiene:
      - `'image'`: `torch.Tensor` de forma `[B, 3, H, W]` en punto flotante.
      - `'point_cloud'`: `torch.Tensor` de forma `[B, N_3D, 3]` en punto flotante.
      - `'gt_mask'`: `torch.Tensor` booleano de forma `[B, N_2D, N_3D]`.
    - `optimizer`: Instancia del optimizador AdamW configurado.
    - `scheduler`: Instancia del planificador de tasa de aprendizaje.
    - `loss_fn`: Instancia de `CircleLossWithSelfPacedWeighting`.
    - `device`: Dispositivo destino (`torch.device("cpu")` o `torch.device("cuda")`).
  - **Flujo Interno de Ejecución**:
    1. Activar el modo de entrenamiento en el modelo: `model.train()`. Asegurar que las capas del backbone DINOv2 que se requiera mantener congeladas de forma permanente no se actualicen en modo entrenamiento (ej. congelando estadísticos de BatchNorm si aplica).
    2. Transferir los tensores del lote (`batch`) al dispositivo especificado.
    3. Reiniciar gradientes del optimizador: `optimizer.zero_grad()`.
    4. Ejecutar el paso forward del modelo para obtener descriptores comunes 2D (`feat_2d` de forma `[B, N_2D, C]`) y descriptores comunes 3D (`feat_3d` de forma `[B, N_3D, C]`). Ambos conjuntos de descriptores deben salir normalizados $L_2$ en la dimensión $C$.
    5. Evaluar la función de pérdida `loss = loss_fn(feat_2d, feat_3d, batch['gt_mask'])`.
    6. Ejecutar el paso backward: `loss.backward()`.
    7. Actualizar parámetros con `optimizer.step()`.
    8. Actualizar tasa de aprendizaje llamando a `scheduler.step()`.
    9. Construir y retornar un diccionario de estadísticas de tipo `Dict[str, float]` con el siguiente formato:
       `{'loss': float(loss.item()), 'lr': float(scheduler.get_last_lr()[0])}`.
- **Criterios de Aceptación**:
  - La función realiza la actualización de parámetros en el dispositivo indicado y devuelve el diccionario de métricas.
  - La llamada a `train_step` no produce fugas de memoria en la GPU (limpiar referencias si es necesario y usar `.item()` en tensores escalares para evitar retener el grafo).

---

### T-TRAIN-04: Diseño y ejecución de pruebas unitarias para el paso de entrenamiento

- **ID de la Tarea**: `T-TRAIN-04`
- **Título**: Diseño y ejecución de pruebas unitarias para `train_step`
- **Descripción**: 
  Implementar pruebas unitarias utilizando mocks y modelos de juguete de PyTorch para verificar que la función del paso de entrenamiento orqueste de manera correcta y secuencial todas las operaciones y modifique únicamente los pesos entrenables del modelo.
- **Requisitos Técnicos Duros**:
  - **Ubicación del archivo**: `tests/training/test_train_step.py`
  - **Casos de prueba obligatorios**:
    1. **Verificación de Llamadas (`test_train_step_orchestration`)**:
       - Utilizar `unittest.mock` para crear versiones simuladas de `model`, `optimizer`, `scheduler` y `loss_fn`.
       - Ejecutar `train_step`.
       - Verificar que se invoquen los métodos en el orden requerido: `model.train()`, `optimizer.zero_grad()`, paso forward en `model`, llamada a `loss_fn`, `loss.backward()`, `optimizer.step()`, y `scheduler.step()`.
    2. **Actualización Efectiva de Pesos (`test_weight_updates`)**:
       - Construir un modelo real en miniatura que tenga dos módulos lineales: `conv_frozen` (congelado con `requires_grad=False`) y `conv_trainable` (entrenable con `requires_grad=True`).
       - Instanciar un optimizador y planificador para este mini-modelo.
       - Almacenar copias profundas (`copy.deepcopy`) de los pesos de ambos módulos.
       - Ejecutar un paso simulado con `train_step` utilizando entradas sintéticas.
       - Comparar pesos antes y después de la ejecución:
         - Comprobar que los pesos de `conv_frozen` sean exactamente iguales.
         - Comprobar que los pesos de `conv_trainable` difieran significativamente de sus valores iniciales.
- **Criterios de Aceptación**:
  - Las pruebas de integración deben correr exitosamente con `pytest tests/training/test_train_step.py`.
  - La prueba de actualización de pesos debe aislar el efecto de la retropropagación en el subgrafo de parámetros activos.
