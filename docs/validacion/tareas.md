# Lista de Tareas Estructuradas: Validación y Estrategia Sim-to-Real (GSCA)

Este documento define la lista detallada y secuencial de tareas atómicas para el módulo de **Validación y Estrategia Sim-to-Real** del proyecto GSCA (Geo-Structural Cross-Attention). Cada tarea representa una unidad de trabajo autocontenida que puede asignarse directamente a un desarrollador.

---

## Índice de Tareas
- [T-VAL-01: Degradación Geométrica de Nubes de Puntos (3D)](#t-val-01-degradación-geométrica-de-nubes-de-puntos-3d)
- [T-VAL-02: Generador del Prior de Pose (6-DoF)](#t-val-02-generador-del-prior-de-pose-6-dof)
- [T-VAL-03: Degradación Visual Físicamente Consistente (2D)](#t-val-03-degradación-visual-físicamente-consistente-2d)
- [T-VAL-04: Pipeline de Evaluación de Pose y Correspondencias](#t-val-04-pipeline-de-evaluación-de-pose-y-correspondencias)
- [T-VAL-05: Suite de Pruebas Unitarias del Módulo de Validación](#t-val-05-suite-de-pruebas-unitarias-del-módulo-de-validación)

---

### T-VAL-01: Degradación Geométrica de Nubes de Puntos (3D)

* **Título**: Implementación del Componente de Degradación Geométrica 3D
* **Descripción técnica**:
  Desarrollar la función `degrade_point_cloud` para aplicar degradaciones sintéticas a la nube de puntos 3D de entrada. Esta función modela la pérdida de densidad y el ruido de medición propios de los escaneos LiDAR y fotogramétricos en campo, permitiendo entrenar al modelo de alineamiento en condiciones de menor calidad geométrica. La degradación consiste en dos etapas secuenciales vectorizadas por lote (batch):
  1. Submuestreo estocástico uniforme de los puntos según una proporción dada.
  2. Inyección de ruido Gaussiano isotrópico a los puntos resultantes.
* **Requisitos técnicos duros**:
  * **Firma exacta de la función**:
    ```python
    def degrade_point_cloud(
        point_cloud: torch.Tensor, 
        noise_std: float, 
        downsample_ratio: float
    ) -> torch.Tensor:
    ```
  * **Tipos y dimensiones**:
    * `point_cloud`: `torch.Tensor` de dimensión `[B, N, 3]`, tipo de datos `torch.float32`. Representa las coordenadas $(x, y, z)$ en metros.
    * `noise_std`: `float` que representa la desviación estándar del ruido Gaussiano (debe validarse que `noise_std >= 0.0`).
    * `downsample_ratio`: `float` en el rango semi-abierto `(0.0, 1.0]` que representa la proporción de puntos a conservar de la nube original.
    * **Salida**: `torch.Tensor` de dimensión `[B, N_degraded, 3]`, tipo `torch.float32`, donde $N_{degraded} = \text{int}(N \times \text{downsample\_ratio})$.
  * **Control de excepciones**:
    * Lanzar `ValueError` si `downsample_ratio` no pertenece al intervalo `(0.0, 1.0]`.
    * Lanzar `ValueError` si `noise_std < 0.0`.
    * Lanzar `TypeError` si `point_cloud` no es un `torch.Tensor`.
  * **Restricciones de rendimiento y hardware**:
    * La función no debe alterar el tensor original `point_cloud` (no mutar in-place).
    * Todas las operaciones (indexación y generación de ruido) deben realizarse en el dispositivo nativo de la entrada (`point_cloud.device`) para evitar transferencias de memoria CPU-GPU.
* **Criterios de aceptación**:
  * **Verificación de dimensiones**: Para una entrada de tamaño `[B, N, 3]`, la salida tiene tamaño exacto `[B, int(N * downsample_ratio), 3]`.
  * **Invariancia de dispositivo**: El tensor devuelto reside en el mismo dispositivo (`cpu`, `cuda`) y mantiene el tipo de datos (`torch.float32`) que el de entrada.
  * **Consistencia estadística**: Al aplicar la función con `downsample_ratio=1.0` y `noise_std=0.03` a una nube inicializada en cero, la desviación estándar empírica de la nube resultante en cada dimensión debe situarse en el intervalo $[0.028, 0.032]$ y la media en el intervalo $[-0.001, 0.001]$.

---

### T-VAL-02: Generador del Prior de Pose (6-DoF)

* **Título**: Implementación del Generador de Perturbaciones Estocásticas de Pose
* **Descripción técnica**:
  Desarrollar la función `synthesize_pose_prior` para generar una pose de cámara inicial perturbada $\mathbf{T}_{prior}$ a partir de la pose real de referencia $\mathbf{T}_{gt}$. Esta función simula la imprecisión del sensor portátil (GPS/IMU) usado por un geólogo en campo, definiendo una pose aproximada a partir de la cual el algoritmo GSCA debe estimar el alineamiento fino. La perturbación se compone de un desplazamiento espacial uniforme y de rotaciones de Euler aleatorias sobre los tres ejes de la cámara.
* **Requisitos técnicos duros**:
  * **Firma exacta de la función**:
    ```python
    def synthesize_pose_prior(
        pose_gt: torch.Tensor, 
        max_trans: float = 1.0, 
        max_rot_deg: float = 5.0
    ) -> torch.Tensor:
    ```
  * **Tipos y dimensiones**:
    * `pose_gt`: `torch.Tensor` de dimensión `[B, 4, 4]`, tipo `torch.float32`. Representa la matriz de transformación homogénea ground truth de la cámara al mundo.
    * `max_trans`: `float` no negativo que indica el desplazamiento uniforme máximo permitido en metros por eje (default: `1.0` m).
    * `max_rot_deg`: `float` no negativo que define la magnitud máxima en grados de la perturbación angular de Euler (default: `5.0` grados).
    * **Salida**: `torch.Tensor` de dimensión `[B, 4, 4]`, tipo `torch.float32`, que representa la pose perturbada $\mathbf{T}_{prior}$.
  * **Control de excepciones**:
    * Lanzar `ValueError` si `max_trans < 0.0` o `max_rot_deg < 0.0`.
    * Lanzar `ValueError` si `pose_gt` no tiene forma `[B, 4, 4]`.
  * **Detalles matemáticos de la implementación**:
    * Convertir `max_rot_deg` a radianes: $max\_rot\_rad = max\_rot\_deg \times (\pi / 180.0)$.
    * Para cada elemento del lote $b$, generar una rotación de perturbación $\mathbf{dR}$ a partir de tres ángulos de Euler independientes $r_x, r_y, r_z$ muestreados de manera uniforme en $[-max\_rot\_rad, max\_rot\_rad]$.
    * Generar una traslación de perturbación $dt$ de tamaño `[3, 1]` con valores muestreados uniformemente en $[-max\_trans, max\_trans]$.
    * Componer la transformación homogénea de perturbación $\mathbf{dT} = \begin{bmatrix} \mathbf{dR} & dt \\ \mathbf{0}_{1\times 3} & 1 \end{bmatrix}$.
    * Aplicar la perturbación multiplicando por la derecha: $\mathbf{T}_{prior} = \mathbf{T}_{gt} \mathbf{dT}$.
* **Criterios de aceptación**:
  * **Verificación de dimensiones**: El tensor de salida es de forma `[B, 4, 4]` y tipo `torch.float32`.
  * **Restricción espacial**: Para cualquier muestra, la diferencia en traslación $\|\mathbf{t}_{prior} - \mathbf{t}_{gt}\|_\infty$ debe ser estrictamente menor o igual a `max_trans`.
  * **Restricción angular**: El error de rotación geodésica entre $\mathbf{R}_{prior}$ y $\mathbf{R}_{gt}$ (calculado como $\theta = \arccos(\text{clamp}(\frac{\text{trace}(\mathbf{R}_{prior}\mathbf{R}_{gt}^T) - 1}{2}, -1.0, 1.0))$) debe ser consistente con la composición de las rotaciones de Euler aleatorias agregadas y no superar el límite matemático impuesto por los límites angulares.

---

### T-VAL-03: Degradación Visual Físicamente Consistente (2D)

* **Título**: Implementación del Módulo de Degradación Visual 2D
* **Descripción técnica**:
  Desarrollar la función `apply_visual_degradations` encargada de inyectar variaciones de iluminación solar y degradación geológica a las imágenes sintéticas RGB. Esto simula las condiciones variables del sol (azimut y elevación), la meteorización química (que altera el albedo original de la roca) y las micro-fracturas o rugosidad de la superficie (que alteran el mapa de normales).
* **Requisitos técnicos duros**:
  * **Firma exacta de la función**:
    ```python
    def apply_visual_degradations(
        image: torch.Tensor, 
        normal_map: torch.Tensor, 
        albedo_map: torch.Tensor, 
        sun_azimuth: torch.Tensor, 
        sun_elevation: torch.Tensor, 
        roughness_factor: float
    ) -> torch.Tensor:
    ```
  * **Tipos y dimensiones**:
    * `image`: `torch.Tensor` de dimensión `[B, 3, H, W]`, tipo `torch.float32`, valores en rango `[0.0, 1.0]`.
    * `normal_map`: `torch.Tensor` de dimensión `[B, 3, H, W]`, tipo `torch.float32`, vectores normalizados en rango `[-1.0, 1.0]`.
    * `albedo_map`: `torch.Tensor` de dimensión `[B, 3, H, W]`, tipo `torch.float32`, valores en rango `[0.0, 1.0]`.
    * `sun_azimuth`: `torch.Tensor` de dimensión `[B]`, tipo `torch.float32`, ángulos en grados `[0.0, 360.0]`.
    * `sun_elevation`: `torch.Tensor` de dimensión `[B]`, tipo `torch.float32`, ángulos en grados `[-90.0, 90.0]`.
    * `roughness_factor`: `float` no negativo que indica el nivel de rugosidad fractal a aplicar.
    * **Salida**: `torch.Tensor` de dimensión `[B, 3, H, W]`, tipo `torch.float32`.
  * **Fórmula e implementación física**:
    * Obtener el vector unitario de luz solar $L$ en coordenadas de cámara a partir de azimut ($\theta$) y elevación ($\phi$):
      $$L = [\cos(\phi)\sin(\theta), \sin(\phi), \cos(\phi)\cos(\theta)]^T$$
      El tensor resultante debe redimensionarse a `[B, 3, 1, 1]`.
    * Generar ruido fractal (Fractional Brownian Motion, fBm) 2D con una resolución coincidente `[H, W]` por lote.
    * Perturbar el mapa de normales: $normal\_map\_perturbed = normal\_map + roughness\_factor \times fractal\_noise\_normal$.
    * Normalizar las normales a lo largo del canal 1: $normal\_map\_perturbed = \frac{normal\_map\_perturbed}{\|normal\_map\_perturbed\|_2 + \epsilon}$, con $\epsilon = 1e-8$.
    * Calcular el término difuso de Lambert: $lambertian = \max(N_{perturbed} \cdot L, 0.0)$.
    * Alterar el albedo usando ruido fractal multiplicativo: $albedo\_weathered = albedo\_map \times (1.0 - 0.2 \times fractal\_noise\_albedo)$.
    * Componer la salida: $image\_degraded = albedo\_weathered \times lambertian$.
    * Aplicar un balance de blancos estocástico multiplicando cada canal RGB por factores aleatorios en el rango $[0.95, 1.05]$.
    * Recortar la salida a `[0.0, 1.0]` usando `torch.clamp`.
  * **Control de excepciones**:
    * Lanzar `ValueError` si `roughness_factor < 0.0`.
    * Lanzar `ValueError` si las formas espaciales `[H, W]` de `image`, `normal_map` y `albedo_map` no coinciden entre sí.
* **Criterios de aceptación**:
  * **Conservación de dimensiones y rango**: La salida debe tener dimensiones exactas `[B, 3, H, W]` con valores restringidos estrictamente en `[0.0, 1.0]`.
  * **Propiedad unitaria de normales**: Al evaluar el mapa de normales perturbado, la norma Euclidiana para cada píxel individual debe ser exactamente $1.0 \pm 1e-6$ en todas las posiciones espaciales.
  * **Estabilidad numérica**: No se deben generar valores no numéricos (`NaN` o `Inf`) debido a divisiones por cero al normalizar las normales perturbadas.

---

### T-VAL-04: Pipeline de Evaluación de Pose y Correspondencias

* **Título**: Implementación del Pipeline de Métricas de Alineamiento
* **Descripción técnica**:
  Desarrollar la función `evaluate_alignment` que calcula el error geométrico entre las poses estimadas $\mathbf{T}_{est}$ (provistas por RANSAC + PnP) y las de referencia $\mathbf{T}_{gt}$, además del error de reproyección de las correspondencias inliers y la tasa de acierto integrada (AUC) bajo tres umbrales de tolerancia.
* **Requisitos técnicos duros**:
  * **Firma exacta de la función**:
    ```python
    def evaluate_alignment(
        pose_est: torch.Tensor, 
        pose_gt: torch.Tensor, 
        correspondences: Dict[str, torch.Tensor], 
        intrinsics_K: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, float]]]:
    ```
  * **Tipos y dimensiones**:
    * `pose_est`: `torch.Tensor` de dimensión `[B, 4, 4]`, tipo `torch.float32`.
    * `pose_gt`: `torch.Tensor` de dimensión `[B, 4, 4]`, tipo `torch.float32`.
    * `correspondences`: `Dict` con las claves:
      * `"pts_2d"`: `torch.Tensor` de dimensión `[B, M, 2]`, tipo `torch.float32` (puntos clave en coordenadas de píxeles).
      * `"pts_3d"`: `torch.Tensor` de dimensión `[B, M, 3]`, tipo `torch.float32` (puntos de la nube de puntos 3D).
      * `"inlier_mask"`: `torch.Tensor` de dimensión `[B, M]`, tipo `torch.bool`.
    * `intrinsics_K`: `torch.Tensor` de dimensión `[B, 3, 3]`, tipo `torch.float32` (matrices intrínsecas de la cámara).
    * **Salida**: `Dict` que contiene:
      * `"translation_error"`: `torch.Tensor` de dimensión `[B]` (error en metros).
      * `"rotation_error"`: `torch.Tensor` de dimensión `[B]` (error geodésico en grados).
      * `"auc_pose"`: `Dict[str, float]` con claves `"easy"`, `"medium"` y `"strict"`.
      * `"inlier_ratio"`: `torch.Tensor` de dimensión `[B]` (proporción de inliers).
      * `"reprojection_error"`: `torch.Tensor` de dimensión `[B]` (promedio del error de reproyección en píxeles para los inliers).
  * **Fórmulas de error**:
    * Error de traslación: $e_{trans} = \|\mathbf{t}_{est} - \mathbf{t}_{gt}\|_2$.
    * Error de rotación: $e_{rot} = \arccos\left(\text{clamp}\left(\frac{\text{trace}(\mathbf{R}_{est}\mathbf{R}_{gt}^T) - 1.0}{2.0}, -1.0, 1.0\right)\right) \times \frac{180.0}{\pi}$.
    * Umbrales de precisión para el cálculo de la tasa de acierto:
      * **Fácil (Easy)**: $e_{trans} \le 0.20$ m y $e_{rot} \le 10^\circ$.
      * **Medio (Medium)**: $e_{trans} \le 0.10$ m y $e_{rot} \le 5^\circ$.
      * **Estricto (Strict)**: $e_{trans} \le 0.05$ m y $e_{rot} \le 1^\circ$.
    * AUC de Pose: Calcular la proporción acumulada de poses que satisfacen los umbrales anteriores e integrar la curva resultante por medio de la regla trapezoidal.
    * Error de reproyección: Proyectar los puntos 3D inliers (donde `inlier_mask` es verdadero) usando la pose de cámara estimada $\mathbf{T}_{est}$ y la matriz intrínseca $\mathbf{K}$:
      $$\mathbf{p}_{proj} = \mathbf{K} \cdot (\mathbf{R}_{est} \cdot \mathbf{P}_{3d} + \mathbf{t}_{est})$$
      y evaluar la distancia Euclidiana media respecto a las coordenadas reales de píxeles `"pts_2d"`.
  * **Control de excepciones**:
    * Manejar de forma segura el caso de "cero inliers" en `inlier_mask` (donde todos son falso), evitando divisiones por cero en el cálculo del error de reproyección (debería retornar un error de `0.0` o `NaN` controlado en tal caso).
* **Criterios de aceptación**:
  * **Integridad del tipo de datos**: La salida debe tener las claves exactas y retornar tensores del tamaño correspondiente al batch $B$.
  * **Prueba analítica de alineamiento**: Ante una pose estimada igual a la real, los errores de rotación y traslación calculados deben ser exactamente `0.0` con tolerancia float.
  * **Comportamiento del AUC**:
    * Si el 100% de las muestras del lote tienen un error inferior a $5$ cm y $1^\circ$, el AUC para todos los niveles (`easy`, `medium`, `strict`) debe reportarse como `1.0`.
    * Si el 100% de las muestras tienen un error mayor a $20$ cm y $10^\circ$, el AUC para todos los niveles debe reportarse como `0.0`.

---

### T-VAL-05: Suite de Pruebas Unitarias del Módulo de Validación

* **Título**: Implementación de Pruebas Unitarias con pytest
* **Descripción técnica**:
  Desarrollar un conjunto exhaustivo de pruebas unitarias automatizadas para asegurar el correcto comportamiento matemático, la consistencia estadística y la estabilidad de tipo de datos en todos los componentes del módulo de validación. Las pruebas deben ser independientes y abarcar todos los casos límite y analíticos definidos en la sección 5 del Plan de Validación.
* **Requisitos técnicos duros**:
  * **Archivo destino**: Las pruebas deben guardarse estrictamente en la ubicación `/tests/test_validation_pipeline.py`.
  * **Casos de prueba a implementar**:
    1. `test_pose_perturbation_limits`: Verificar que para 1000 iteraciones con pose ground truth de identidad, la traslación del prior esté estrictamente dentro de $[-\text{max\_trans}, \text{max\_trans}]$ por eje y el ángulo de rotación residual esté acotado de manera adecuada.
    2. `test_point_cloud_degradation_statistics`: Evaluar una nube de puntos inicializada en cero de tamaño `[1, 10000, 3]` degradada con `downsample_ratio=0.5` y `noise_std=0.03`. Afirmar que el tamaño de salida sea exactamente `[1, 5000, 3]`, que la desviación estándar empírica esté en el intervalo $[0.028, 0.032]$ y que la media de la perturbación se mantenga en $[-0.001, 0.001]$.
    3. `test_normal_map_conservation`: Validar que tras aplicar rugosidad con `apply_visual_degradations` sobre un mapa de normales uniforme `[0, 0, 1]`, la norma Euclidiana de los vectores en cada píxel resultante sea exactamente $1.0 \pm 1e-6$.
    4. `test_pose_error_metrics_correctness`: Probar un caso analítico donde la pose estimada represente una rotación de $90^\circ$ sobre el eje Z con una traslación de $[1.5, 0.0, 0.0]$ metros respecto a la pose real (Identidad). Validar que la traslación calculada sea exactamente $1.5 \pm 1e-5$ metros y la rotación sea $90.0^\circ \pm 1e-5$.
    5. `test_auc_calculation_control`: Verificar el cálculo de los tres niveles del AUC bajo condiciones extremas:
       - Control Positivo: Errores de traslación < $5$ cm y rotación < $1^\circ$ en todo el lote (debe dar AUC = 1.0).
       - Control Negativo: Errores de traslación = $1.0$ metro y rotación = $15^\circ$ en todo el lote (debe dar AUC = 0.0).
  * **Tecnología**: Uso obligatorio de `pytest` como framework de ejecución.
* **Criterios de aceptación**:
  * **Ejecución exitosa**: Ejecutar `pytest tests/test_validation_pipeline.py` y lograr que el 100% de las aserciones pasen sin generar advertencias críticas o errores de ejecución.
  * **Tolerancias estrictas**: Utilizar métodos de aserción numéricos con tolerancia precisa (`torch.allclose`, `math.isclose`) con una precisión de al menos $1e-5$.
