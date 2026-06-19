# 02. Rama Geométrica (3D): Representación de Estructuras con DGCNN y EdgeConv

La rama geométrica (3D) procesa nubes de puntos LiDAR o mallas geológicas estructuradas para obtener descriptores topológicos densos $\mathbf{f}_j^{3D}$ que capturen las discontinuidades invariantes de la roca.

---

## 1. Limitaciones de PointNet y la Bolsa de Elementos

Los baselines tradicionales de aprendizaje profundo en 3D, como **PointNet**, procesan nubes de puntos de manera aislada utilizando perceptrones multicapa (MLP) compartidos seguidos por una operación de pooling simétrica global (usualmente `max-pooling`).

### El Problema de PointNet en Geología
Al operar en cada punto individualmente antes de la agregación global, PointNet trata la nube como una "bolsa de elementos" independiente. Ignora las relaciones locales y la correlación geométrica entre puntos vecinos. En geología, esto es inaceptable porque la identidad geológica de un punto (ej. si pertenece a un plano de falla, a una estría de deslizamiento o a un contacto estratigráfico) está definida **intrínsecamente por su vecindario geométrico y su orientación relativa**.

---

## 2. Dynamic Graph CNN (DGCNN)

Para incorporar la topología y conectividad del macizo rocoso, la arquitectura adopta **DGCNN** (Dynamic Graph CNN).

A diferencia de PointNet, DGCNN construye y actualiza de manera dinámica grafos de vecindad locales en el espacio de características latentes de cada capa de la red. Esto permite:
* **Conectividad Semántica**: Agrupar de manera compacta puntos que, aunque estén físicamente distantes en la nube de puntos euclidiana original, comparten propiedades estructurales y orientaciones similares en el espacio latente.
* **Propagación Local-Global**: Mantener la coherencia topológica y capturar las transiciones suaves e abruptas del terreno.

---

## 3. El Operador EdgeConv

El núcleo matemático de DGCNN es el operador **EdgeConv**, que extrae características locales relacionando cada punto con sus vecinos más cercanos.

### Formulación Matemática
Para cada punto $p_i = (x_i, y_i, z_i) \in \mathbb{R}^3$, se construye un grafo dirigido $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ utilizando sus $k$-vecinos más cercanos ($k$-NN) en el espacio de características actual.

La característica de arista (edge feature) $e_{ij}$ entre el punto central $p_i$ y su vecino $p_j$ se define como:

$$e_{ij} = h_{\Theta}(p_i, p_j - p_i)$$

Donde:
* $h_{\Theta}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^{d'}$ es una función no lineal parametrizada por un perceptrón multicapa (MLP).
* Al pasar la tupla $(p_i, p_j - p_i)$, la red codifica tanto la información del punto local como la diferencia vectorial relativa (geometría y dirección del vecindario).

Finalmente, las características de arista de todos los vecinos se agregan mediante una operación simétrica (usualmente el valor máximo) para obtener la nueva representación del punto $p'_i$:

$$p'_i = \max_{j : (i, j) \in \mathcal{E}} e_{ij}$$

---

## 4. Estructura Multi-Escala y Dilatación Estructural

Las rocas tienen una naturaleza fractal. Para abordar esta jerarquía multiescala:
1. Se configuran múltiples operadores EdgeConv con **radios de agregación ($k$) progresivamente mayores** y factores de dilatación en el grafo.
2. Esto captura desde la micro-rugosidad de fallas menores ($k$ pequeño) hasta la estratigrafía macroscópica de capas sedimentarias ($k$ grande).
3. Las características de todas las escalas se concatenan al final para formar el descriptor geométrico denso final $\mathbf{f}_j^{3D} \in \mathbb{R}^{C}$.

---

## 5. Guía de Implementación en PyTorch Geometric

En Python, la biblioteca recomendada para la implementación es **PyTorch Geometric (PyG)**, que ya proporciona implementaciones altamente optimizadas del operador `DynamicEdgeConv`.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

class GeometricFeatureExtractor(nn.Module):
    def __init__(self, out_channels: int = 256, k: int = 20):
        super().__init__()
        self.k = k
        
        # Bloques de Convolución en Grafo Dinámico (DynamicEdgeConv)
        # Cada capa calcula k-NN dinámicamente en el espacio latente
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(3 * 2, 64), nn.BatchNorm1d(64), nn.ReLU()),
            k=k,
            aggr='max'
        )
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(64 * 2, 128), nn.BatchNorm1d(128), nn.ReLU()),
            k=k,
            aggr='max'
        )
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(128 * 2, 256), nn.BatchNorm1d(256), nn.ReLU()),
            k=k,
            aggr='max'
        )
        
        # Capas de agregación final y proyección al espacio latente común Z
        self.proj = nn.Sequential(
            nn.Linear(64 + 128 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )

    def forward(self, pos: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        # pos: [N, 3] tensor de coordenadas 3D (x, y, z)
        # batch: [N] tensor que identifica el batch de cada punto en PyG
        
        # Extracción de características a múltiples escalas
        x1 = self.conv1(pos, batch)      # [N, 64]
        x2 = self.conv2(x1, batch)       # [N, 128]
        x3 = self.conv3(x2, batch)       # [N, 256]
        
        # Concatenación multi-escala (Skip-connections para capturar jerarquía fractal)
        x_concat = torch.cat([x1, x2, x3], dim=-1)  # [N, 448]
        
        # Proyección final
        feat_3d = self.proj(x_concat)    # [N, out_channels]
        
        # Normalización L2 para espacio métrico común
        feat_3d = nn.functional.normalize(feat_3d, p=2, dim=-1)
        return feat_3d
```
