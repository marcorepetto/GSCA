# Lista de Tareas: Rama Geométrica (3D) - GSCA

Este documento desglosa el plan de desarrollo de la **Rama Geométrica (3D)** de la arquitectura **GSCA (Geo-Structural Cross-Attention)** en tareas atómicas, secuenciales y auto-contenidas. Cada tarea define requisitos técnicos estrictos y criterios de aceptación inequívocos para guiar al desarrollador.

---

## Dependencias de Software Requeridas
* **PyTorch** $\ge$ 2.0.0
* **PyTorch Geometric (PyG)** $\ge$ 2.3.0 (con soporte CUDA compilado para `DynamicEdgeConv` y `knn`)
* **CUDA Toolkit** (versión coincidente con la instalación de PyTorch)

---

## Tabla de Tareas

| ID de Tarea | Título | Dependencias |
| :--- | :--- | :--- |
| **T-GEOM-01** | Configuración de Dependencias y Validación de Entorno | Ninguna |
| **T-GEOM-02** | Definición del Constructor del Extractor de Características | T-GEOM-01 |
| **T-GEOM-03** | Implementación del Paso Forward: Capas de Grafo Dinámico | T-GEOM-02 |
| **T-GEOM-04** | Implementación del Paso Forward: Fusión, Proyección y Normalización L2 | T-GEOM-03 |
| **T-GEOM-05** | Pruebas Unitarias: Invarianza a Permutación y Traslación | T-GEOM-04 |
| **T-GEOM-06** | Pruebas Unitarias: Consistencia y Aislamiento del Batch | T-GEOM-04 |
| **T-GEOM-07** | Pruebas Unitarias: Integridad del Flujo de Gradiente (Backward Pass) | T-GEOM-04 |

---

## Desglose de Tareas Detallado

### T-GEOM-01: Configuración de Dependencias y Validación de Entorno
* **ID**: `T-GEOM-01`
* **Título**: Configuración de Dependencias y Validación de Entorno
* **Descripción**: Verificar e integrar de forma correcta las dependencias críticas de procesamiento geométrico 3D. Se debe asegurar que las librerías PyTorch y PyTorch Geometric (PyG) estén instaladas con soporte CUDA habilitado para garantizar la viabilidad del operador `DynamicEdgeConv` sobre nubes de puntos densas sin penalización extrema de rendimiento.
* **Requisitos Técnicos Duros**:
  * Implementar un script utilitario o rutina de validación que ejecute aserciones estrictas sobre las versiones del entorno:
    ```python
    import torch
    import torch_geometric
    
    assert torch.__version__ >= "2.0.0", "PyTorch debe ser >= 2.0.0"
    assert torch_geometric.__version__ >= "2.3.0", "PyTorch Geometric debe ser >= 2.3.0"
    assert torch.cuda.is_available(), "Soporte CUDA obligatorio para procesar grafos dinámicos"
    ```
  * Asegurar que se puede importar `DynamicEdgeConv` sin lanzar excepciones de carga dinámica de librerías compartidas (`OSError`/`ImportError` debido a desajustes de versiones CUDA en PyG):
    ```python
    from torch_geometric.nn import DynamicEdgeConv
    ```
* **Criterios de Aceptación**:
  * Ejecución exitosa del script de verificación. El proceso debe retornar un código de estado `0` y confirmar en salida estándar (`stdout`) que CUDA, PyTorch y PyTorch Geometric están acoplados correctamente para operaciones en GPU.

---

### T-GEOM-02: Definición del Constructor del Extractor de Características
* **ID**: `T-GEOM-02`
* **Título**: Definición del Constructor del Extractor de Características (`GeometricFeatureExtractor`)
* **Descripción**: Definir la estructura general y declarar las variables y submódulos de la clase `GeometricFeatureExtractor` en un archivo modular (por ejemplo, `gsca/models/geometric.py`). En el método `__init__`, se deben instanciar las capas de convolución en grafo dinámico (`DynamicEdgeConv`) con sus respectivos perceptrones multicapa (MLPs) de procesamiento local de aristas, además de la red de proyección lineal global.
* **Requisitos Técnicos Duros**:
  * Heredar estrictamente de `torch.nn.Module`.
  * Firma del constructor exacta:
    ```python
    def __init__(self, k: int = 20, out_channels: int = 256):
    ```
  * Módulos requeridos a inicializar internamente:
    1. **MLP 1** (para la primera capa de convolución): `nn.Sequential` que mapea de $6$ dimensiones ($3$ relativas + $3$ absolutas en coordenadas 3D físicas) a $64$ dimensiones. Debe contener: `nn.Linear(6, 64)`, `nn.BatchNorm1d(64)` y `nn.ReLU()`.
    2. **EdgeConv 1 (Micro-escala)**: Instancia de `DynamicEdgeConv` utilizando `MLP 1`, el hiperparámetro `k` y agregación simétrica `aggr="max"`.
    3. **MLP 2** (para la segunda capa de convolución): `nn.Sequential` que mapea de $128$ dimensiones (características $64 \cdot 2$) a $128$ dimensiones. Debe contener: `nn.Linear(128, 128)`, `nn.BatchNorm1d(128)` y `nn.ReLU()`.
    4. **EdgeConv 2 (Media-escala)**: Instancia de `DynamicEdgeConv` utilizando `MLP 2`, el hiperparámetro `k` y agregación `aggr="max"`.
    5. **MLP 3** (para la tercera capa de convolución): `nn.Sequential` que mapea de $256$ dimensiones (características $128 \cdot 2$) a $256$ dimensiones. Debe contener: `nn.Linear(256, 256)`, `nn.BatchNorm1d(256)` y `nn.ReLU()`.
    6. **EdgeConv 3 (Macro-escala)**: Instancia de `DynamicEdgeConv` utilizando `MLP 3`, el hiperparámetro `k` y agregación `aggr="max"`.
    7. **Red de Proyección Lineal**: Mapea la característica concatenada multi-escala ($448$ canales) al espacio de descriptores unitario de dimensión `out_channels` (por defecto $256$). Estructura secuencial requerida: `nn.Linear(448, 512)`, `nn.BatchNorm1d(512)`, `nn.ReLU()` y `nn.Linear(512, out_channels)`.
* **Criterios de Aceptación**:
  * Instanciación exitosa del objeto `extractor = GeometricFeatureExtractor(k=20, out_channels=256)`.
  * Verificación por inspección estructural de que todos los parámetros del modelo se registren correctamente en PyTorch (`model.parameters()` contiene los pesos y sesgos de todas las subcapas lineales y de normalización especificadas).

---

### T-GEOM-03: Implementación del Paso Forward: Capas de Grafo Dinámico
* **ID**: `T-GEOM-03`
* **Título**: Implementación del Paso Forward: Capas de Grafo Dinámico
* **Descripción**: Desarrollar la primera fase de procesamiento en el método `forward`. Este procesa los tensores espaciales y de lote a través de la secuencia de bloques EdgeConv de forma encadenada. Las vecindades de grafos dinámicos en cada capa deben recalcularse usando el espacio latente obtenido de la capa previa, cuidando no mezclar información entre distintas nubes en el minilote.
* **Requisitos Técnicos Duros**:
  * Firma exacta del método:
    ```python
    def forward(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor
    ```
  * **Tipos y Formas Esperadas de las Entradas**:
    * `pos`: `torch.Tensor` de tipo `torch.float32` y forma `[N, 3]`.
    * `batch`: `torch.Tensor` de tipo `torch.int64` y forma `[N]`.
  * **Manejo de Errores e Integridad del Tensor**:
    * Lanzar un `ValueError` con un mensaje descriptivo si `pos` o `batch` no son de tipo `torch.Tensor`.
    * Lanzar un `ValueError` si las dimensiones espaciales no son tridimensionales: `pos.ndim != 2` o `pos.shape[1] != 3`.
    * Lanzar un `ValueError` si las longitudes de los tensores difieren: `pos.shape[0] != batch.shape[0]`.
  * **Flujo Computacional**:
    1. Ejecutar primera convolución espacial local: `x1 = self.conv1(pos, batch)`. Salida intermedia esperada: `[N, 64]`.
    2. Ejecutar segunda convolución dinámica latente: `x2 = self.conv2(x1, batch)`. Salida intermedia esperada: `[N, 128]`.
    3. Ejecutar tercera convolución dinámica latente: `x3 = self.conv3(x2, batch)`. Salida intermedia esperada: `[N, 256]`.
* **Criterios de Aceptación**:
  * La ejecución del flujo de convoluciones debe ejecutarse completamente en GPU sin generar advertencias de desbordamiento de memoria ni errores de dimensiones entre capas consecutivas.

---

### T-GEOM-04: Implementación del Paso Forward: Fusión, Proyección y Normalización L2
* **ID**: `T-GEOM-04`
* **Título**: Implementación del Paso Forward: Fusión, Proyección y Normalización L2
* **Descripción**: Completar el método `forward` consolidando las características extraídas a diferentes escalas de abstracción (skip connections). Concatenar los tensores intermedios del paso anterior, aplicar la proyección lineal global para homogeneizar el tamaño del descriptor con la rama 2D y finalmente normalizar cada vector descriptor a la hiperesfera unitaria ($L_2$) para su uso congruente en la pérdida métrica y el módulo de atención.
* **Requisitos Técnicos Duros**:
  * **Fusión Multi-Escala**: Concatenar los tensores `x1` (`[N, 64]`), `x2` (`[N, 128]`) y `x3` (`[N, 256]`) por la última dimensión:
    ```python
    x_concat = torch.cat([x1, x2, x3], dim=-1)  # Forma esperada: [N, 448]
    ```
  * **Red de Proyección**: Mapear `x_concat` a través de las capas de proyección:
    ```python
    x_proj = self.projection(x_concat)  # Forma esperada: [N, out_channels]
    ```
  * **Normalización Métrica**: Aplicar una proyección de norma unitaria a lo largo del último eje:
    ```python
    feat_3d = torch.nn.functional.normalize(x_proj, p=2, dim=-1)  # Forma esperada: [N, out_channels]
    ```
  * Retornar un tensor de tipo `torch.float32` and forma `[N, out_channels]`.
* **Criterios de Aceptación**:
  * El retorno de la función `forward` debe ser un tensor con forma exacta `[N, out_channels]`.
  * La norma matemática de cada descriptor individual $j$ en el conjunto de salida debe ser exactamente $1.0$ (es decir, $\sum_{c=1}^{C} (f_{j, c})^2 = 1.0$), verificado con una tolerancia numérica de precisión simple menor o igual a $10^{-6}$.

---

### T-GEOM-05: Pruebas Unitarias: Invarianza a Permutación y Traslación
* **ID**: `T-GEOM-05`
* **Título**: Pruebas Unitarias: Invarianza a Permutación y Traslación
* **Descripción**: Escribir e integrar pruebas en la suite del proyecto para validar que el comportamiento interno del extractor geométrico respeta los principios fundamentales de la invariancia espacial. El ordenamiento arbitrario de la lista de puntos de entrada no debe alterar los descriptores resultantes (permutación), y la traslación rígida global de la nube de puntos 3D en el espacio físico no debe variar la codificación semántica (traslación).
* **Requisitos Técnicos Duros**:
  * Las pruebas unitarias deben escribirse utilizando el framework `pytest`.
  * **Verificación de Invariancia a la Permutación**:
    1. Generar una nube de puntos sintética aleatoria `pos_A` de forma `[N, 3]` y un tensor de lote `batch_A` de forma `[N]`.
    2. Generar un vector de índices permutados aleatoriamente `P = torch.randperm(N)`.
    3. Construir la nube y el lote permutado: `pos_B = pos_A[P]`, `batch_B = batch_A[P]`.
    4. Evaluar el modelo en ambos estados: `feat_A = model(pos_A, batch_A)` and `feat_B = model(pos_B, batch_B)`.
    5. Validar que la diferencia numérica sea nula:
       ```python
       torch.allclose(feat_A[P], feat_B, atol=1e-5)
       ```
  * **Verificación de Invariancia a la Traslación Global**:
    1. Generar una nube de puntos sintética `pos`.
    2. Generar un vector de traslación espacial aleatorio `t = torch.randn(1, 3)`.
    3. Construir la nube trasladada: `pos_trans = pos + t`.
    4. Evaluar el modelo en ambos estados: `feat_orig = model(pos, batch)` y `feat_trans = model(pos_trans, batch)`.
    5. Validar la coincidencia exacta de los descriptores resultantes:
       ```python
       torch.allclose(feat_orig, feat_trans, atol=1e-5)
       ```
* **Criterios de Aceptación**:
  * Ejecución exitosa de `pytest` en el suite de pruebas unitarias. Las aserciones matemáticas deben pasar sin generar discrepancias mayores a $10^{-5}$.

---

### T-GEOM-06: Pruebas Unitarias: Consistencia y Aislamiento del Batch
* **ID**: `T-GEOM-06`
* **Título**: Pruebas Unitarias: Consistencia y Aislamiento del Batch
* **Descripción**: Diseñar e implementar pruebas de verificación para confirmar la dimensionalidad de las salidas bajo configuraciones dinámicas y comprobar que no exista filtración de información cruzada entre ejemplos distintos empaquetados en un mismo minilote (aislamiento del lote en los grafos $k$-NN).
* **Requisitos Técnicos Duros**:
  * **Prueba de Consistencia y Normalización**:
    1. Instanciar el extractor con `out_channels = 128` y `out_channels = 256` secuencialmente.
    2. Alimentar nubes de puntos de tamaño arbitrario.
    3. Comprobar que la forma del tensor sea exactamente `[N, out_channels]`.
    4. Comprobar que todas las filas tengan una norma euclidiana estrictamente igual a $1.0$ (tolerancia de error de precisión simple de $10^{-6}$).
  * **Prueba de Aislamiento del Batch**:
    1. Crear dos nubes de puntos distintas y dispares: `pos_1` de tamaño $N_1$ y `pos_2` de tamaño $N_2$.
    2. Procesar `pos_1` de manera aislada con un lote nulo `batch_1 = torch.zeros(N_1, dtype=torch.long)`. Almacenar descriptor `feat_solo_1`.
    3. Empaquetar ambas nubes simulando la paralelización dispersa de PyG:
       ```python
       pos_all = torch.cat([pos_1, pos_2], dim=0)
       batch_all = torch.cat([torch.zeros(N_1, dtype=torch.long), torch.ones(N_2, dtype=torch.long)])
       ```
    4. Procesar el batch unificado: `feat_batch_all = model(pos_all, batch_all)`.
    5. Extraer la sección del primer elemento: `feat_batch_1 = feat_batch_all[:N_1]`.
    6. Asegurar que los descriptores no sufrieron alteración por la presencia de `pos_2` en los cálculos de vecinos latentes:
       ```python
       torch.allclose(feat_solo_1, feat_batch_1, atol=1e-6)
       ```
* **Criterios de Aceptación**:
  * Ejecución satisfactoria de la suite de pruebas. El aislamiento del batch debe ser absoluto; la prueba de aislamiento debe fallar de forma intencionada si el módulo de grafo no hace uso del tensor `batch` para restringir la búsqueda de vecinos $k$-NN.

---

### T-GEOM-07: Pruebas Unitarias: Integridad del Flujo de Gradiente (Backward Pass)
* **ID**: `T-GEOM-07`
* **Título**: Pruebas Unitarias: Integridad del Flujo de Gradiente (Backward Pass)
* **Descripción**: Asegurar la diferenciabilidad matemática completa del pipeline implementado. La prueba debe verificar que no se produzcan interrupciones en el grafo computacional de autograd debido a clonaciones de tensores erróneas, casteo de tipos inconsistentes o inicialización incorrecta de variables auxiliares, y descartar inestabilidades como gradientes nulos o infinitos.
* **Requisitos Técnicos Duros**:
  * Establecer el modelo en modo de entrenamiento: `model.train()`.
  * Permitir gradientes en los tensores de entrada y verificar que todos los parámetros del modelo tengan activo `requires_grad=True`.
  * Realizar un paso forward y definir una función de pérdida escalar diferenciable a partir de la salida (ejemplo: `loss = feat_3d.sum()`).
  * Ejecutar el paso de retropropagación: `loss.backward()`.
  * Recorrer de forma recursiva los parámetros entrenables y aseverar:
    1. El gradiente asociado no es nulo: `assert param.grad is not None` para cada tensor de peso de las capas lineales y de normalización por lotes.
    2. El gradiente es numéricamente estable:
       ```python
       assert not torch.isnan(param.grad).any(), "Gradiente contiene valores NaN"
       assert not torch.isinf(param.grad).any(), "Gradiente contiene valores infinitos"
       ```
* **Criterios de Aceptación**:
  * Las pruebas de estabilidad de gradientes se ejecutan y completan con éxito, asegurando que todos los pesos entrenables participan activamente en el aprendizaje supervisado y que la actualización por descenso de gradiente es matemáticamente viable.
