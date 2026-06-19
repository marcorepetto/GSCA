# 05. Validación y Estrategia Sim-to-Real

La validación y prueba de la arquitectura GSCA aborda la brecha entre simulaciones y datos del mundo real mediante un entorno virtual controlado y técnicas de transferencia robustas.

---

## 1. El Laboratorio Sintético (Unity / OpenGL)

Validar algoritmos de matching 2D-3D en afloramientos reales presenta dos desafíos severos:
1. **Falta de Ground Truth Absoluto**: Los sistemas GPS/IMU portátiles en campo no proveen precisiones sub-centimétricas para la pose 6-DoF de la cámara.
2. **Estocasticidad de la Iluminación**: Es imposible controlar la posición del sol o las nubes para aislar el efecto de la iluminación en el algoritmo.

Para mitigar esto, se propone un pipeline de generación de datos sintéticos en **Unity o OpenGL** que actúa como un entorno experimental controlado.

### Beneficios del Entorno Virtual
* **Aislamiento de Variables**: Permite manipular sistemáticamente el ángulo del sol (acimut y elevación) y la nubosidad para probar la resiliencia a la BRDF.
* **Verdad de Campo Exacta**: Obtención del tensor de pose real $(\mathbf{R}, \mathbf{t})$ sin ruido de medición física.
* **Mapas de Normales de Referencia**: Generación de mapas de normales 2D e información de adyacencia topográfica 3D exacta para supervisar y testear la máscara $\mathbf{M}_{geo}$.

---

## 2. Estrategia de Degradación Sim-to-Real

Para evitar que la red sufra de una **brecha de dominio geométrica (geometric domain gap)** al transferirse a datos reales, los datos sintéticos de entrenamiento se someten a un pipeline de degradación estocástica:

### A. Degradación en la Nube de Puntos (3D)
* **Inyección de Ruido**: Se añade ruido Gaussiano en las tres coordenadas espaciales ($x, y, z$).
* **Submuestreo Estocástico**: Se realiza un submuestreo aleatorio para emular la densidad de puntos irregular y oclusiones propias de los escaneos LiDAR reales en campo.

### B. Degradación en las Imágenes (2D)
Durante el renderizado sintético de las 10,000 imágenes de entrenamiento, se varían estocásticamente tres propiedades físicas:
* **Ángulo de Incidencia Solar**: Variación del acimut y la elevación en un rango completo de $0^\circ$ a $360^\circ$ para forzar invarianza lumínica.
* **Rugosidad Física de la Roca**: Aplicación de mapas de normales perturbados por ruido fractal para simular micro-fracturas.
* **Reflectancia Difusa (Albedo)**: Modificación de los mapas de albedo de textura para simular meteorización química y mineral del afloramiento.

---

## 3. Acceso a Datos Reales y Transferencia Zero-Shot

Para consolidar la viabilidad del proyecto, el modelo entrenado sintéticamente (con $10,000$ imágenes renderizadas) se evalúa directamente, **sin ajuste fino adicional (Zero-Shot)**, en conjuntos de datos reales obtenidos de plataformas públicas:
* **Svalbox**: Modelos 3D de alta resolución y fotogrametría digital de bancos geológicos en Svalbard.
* **eRock**: Repositorio académico de nubes de puntos geológicas.
* **OpenTopography**: Datos de elevación y topografía de gran escala.

Esto demuestra la capacidad de generalización pura del sesgo inductivo geológico inyectado en el modelo.

---

## 4. Síntesis del Prior de Pose $T_{prior}$

Para emular las condiciones en campo, durante las pruebas se introduce una perturbación sintética en la pose real para generar el prior $\mathbf{T}_{prior}$:
* **Traslación**: Se aplica un desplazamiento uniforme en las tres direcciones:
  $$\Delta \mathbf{t} \sim \mathcal{U}(-1\text{m}, 1\text{m})$$
* **Rotación**: Se perturba la orientación $\Delta \mathbf{R}$ utilizando ángulos de Euler aleatorios con una magnitud máxima de $5^\circ$.

---

## 5. Baselines y Métricas de Evaluación

Para cuantificar el desempeño de GSCA, se compara con tres tipos de baselines del estado del arte:

| Categoría de Baseline | Algoritmo | Descripción |
| :--- | :--- | :--- |
| **Local Matching (2D)** | SuperPoint + SuperGlue + EPnP | Basado en gradientes visuales locales; evalúa el límite de representaciones 2D clásicas ante simetrías. |
| **Cross-Domain (2D-3D)** | 2D3D-MVPNet y LCD | Redes duales de aprendizaje de métricas que proyectan subvolúmenes en planos 2D. |
| **End-to-End (Directo)** | EP2P-Loc | Localización directa sin correspondencias de puntos clave (detección de parches y PnP diferenciable). |

### Métricas de Rendimiento
1. **AUC del Error de Pose (Area Under the Curve)**:
   Se mide el error de pose 6-DoF y se calcula el AUC bajo tres umbrales estrictos de tolerancia:
   * Fácil: $(20\text{cm}, 10^\circ)$
   * Medio: $(10\text{cm}, 5^\circ)$
   * Estricto: $(5\text{cm}, 1^\circ)$
2. **Calidad de las Correspondencias**:
   * **Inlier Ratio**: Proporción de correspondencias MNN válidas que satisfacen el modelo geométrico tras pasar por RANSAC.
   * **Error de Reproyección**: Promedio del error de proyección de los puntos 3D emparejados al plano de la imagen (en píxeles).
