# 03. Atención Cruzada Estructural (GSCA) y Matching MNN

El módulo **GSCA (Geo-Structural Cross-Attention)** actúa como un puente cognitivo que utiliza la certidumbre de la estructura geométrica 3D para guiar, filtrar y desambiguar la extracción de descriptores en el espacio visual 2D.

---

## 1. Proyección Geométrica de la Nube de Puntos

Para alinear el dominio 2D y el 3D, primero se proyecta la nube de puntos 3D en el plano de la imagen. Esto requiere una calibración intrínseca de la cámara $\mathbf{K}_{cam}$ y una estimación aproximada (prior) de la pose de la cámara $\mathbf{T}_{prior} = [\mathbf{R}_{prior} | \mathbf{t}_{prior}] \in SE(3)$:

$$\lambda_j \begin{bmatrix} u_j \\ vj \\ 1 \end{bmatrix} = \mathbf{K}_{cam} (\mathbf{R}_{prior} \mathbf{p}_j + \mathbf{t}_{prior})$$

Donde:
* $\mathbf{p}_j = (x_j, y_j, z_j)^T$ es la coordenada física 3D del punto $j$ en el modelo global.
* $\lambda_j$ es la profundidad del punto proyectado respecto al plano de la cámara.
* $(u_j, v_j)$ son las coordenadas de píxel proyectadas en la imagen 2D.

---

## 2. Mecanismo de Atención Cruzada Regulada

La atención cruzada tradicional calcula la afinidad entre consultas (Queries, derivadas de la imagen) y claves (Keys, derivadas de la nube de puntos). Para evitar correspondencias falsas en geometrías altamente repetitivas, GSCA introduce una máscara estructural atenuante, $\mathbf{M}_{geo} \in \mathbb{R}^{HW \times N}$:

$$\mathbf{A}_{cross} = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}_{keys}^T}{\sqrt{C}} + \mathbf{M}_{geo} \right)$$

Donde:
* $\mathbf{Q} \in \mathbb{R}^{HW \times C}$ son las consultas del mapa de descriptores visuales 2D.
* $\mathbf{K}_{keys} \in \mathbb{R}^{N \times C}$ son las claves obtenidas del descriptor geométrico 3D.
* $C$ es la dimensión del espacio latente común.

### Máscara Contextual Estructural $M_{geo}$
La máscara $\mathbf{M}_{geo}(i, j)$ filtra las correspondencias inverosímiles penalizando distancias euclidianas proyectadas excesivas y desalineamientos en las normales de superficie:

$$\mathbf{M}_{geo}(i, j) = \begin{cases} 0 & \text{si } \|(u_i, v_i) - (u_j, v_j)\|_2 \le \delta \quad \text{y} \quad \langle \mathbf{n}_i, \mathbf{n}_j \rangle \ge \tau \\ -\infty & \text{en caso contrario} \end{cases}$$

Explicación de las condiciones:
1. **Filtro de Distancia Proyectada ($\delta$)**: Limita la búsqueda a un radio $\delta$ en píxeles alrededor de la proyección teórica. $\delta$ se calibra basándose en la incertidumbre del prior de pose (asumiendo un error máximo de traslación $< 1m$).
2. **Consistencia de Orientación / Coplanaidad ($\tau$)**: Compara la orientación del vector normal estimado en la imagen 2D ($\mathbf{n}_i$) con la normal geométrica en 3D ($\mathbf{n}_j$). El producto punto $\langle \mathbf{n}_i, \mathbf{n}_j \rangle$ debe ser mayor que el umbral $\tau$. Las normales 2D se asumen pre-calculadas por un modelo auxiliar independiente y su error residual se absorbe mediante un umbral suave $\tau$.
3. **Penalización Máxima ($-\infty$)**: Al aplicar softmax, los pesos de atención para las posiciones con $-\infty$ colapsan exactamente a 0, aislando el flujo de gradientes y limitando el matching a vecindarios geométricos físicamente plausibles.

---

## 3. Mutual Nearest Neighbors (MNN)

Una vez filtradas y actualizadas las características por el mapa de atención cruzada, el emparejamiento denso final se resuelve mediante una búsqueda de vecinos más cercanos mutuos en el espacio latente. 

Se establece una correspondencia sólida entre el píxel $i$ y el punto 3D $j$ si y solo si cumplen la condición de bidireccionalidad:

$$j = \arg\max_{k} \text{sim}(\mathbf{f}_i^{2D}, \mathbf{f}_k^{3D}) \quad \text{y} \quad i = \arg\max_{l} \text{sim}(\mathbf{f}_l^{2D}, \mathbf{f}_j^{3D})$$

Donde la función de similitud es la similitud coseno:

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

---

## 4. Guía de Implementación en PyTorch

El siguiente fragmento ilustra cómo construir la atención cruzada enmascarada y el matching MNN:

```python
import torch
import torch.nn as nn

class GeoStructuralCrossAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.scale = channels ** 0.5
        
        # Proyecciones lineales para la atención cruzada
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        
    def forward(self, 
                feat_2d: torch.Tensor,   # [B, HW, C]
                feat_3d: torch.Tensor,   # [B, N, C]
                coords_2d: torch.Tensor, # [B, HW, 2] (coordenadas de píxel i)
                proj_coords: torch.Tensor, # [B, N, 2] (coordenadas proyectadas u_j, v_j)
                normals_2d: torch.Tensor,  # [B, HW, 3]
                normals_3d: torch.Tensor,  # [B, N, 3]
                delta: float = 30.0,     # Radio de tolerancia en píxeles
                tau: float = 0.5         # Umbral de coplanaridad
                ) -> torch.Tensor:
        B, HW, _ = feat_2d.shape
        _, N, _ = feat_3d.shape
        
        # 1. Proyecciones
        Q = self.q_proj(feat_2d) # [B, HW, C]
        K = self.k_proj(feat_3d) # [B, N, C]
        V = self.v_proj(feat_3d) # [B, N, C]
        
        # 2. Calcular afinidad cruda
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale # [B, HW, N]
        
        # 3. Construir la Máscara M_geo
        # 3.1 Distancia euclidiana proyectada
        # coords_2d: [B, HW, 1, 2], proj_coords: [B, 1, N, 2]
        dist = torch.cdist(coords_2d, proj_coords, p=2) # [B, HW, N]
        dist_mask = dist > delta
        
        # 3.2 Producto punto de normales (Coplanaidad)
        # normals_2d: [B, HW, 3], normals_3d: [B, N, 3]
        cos_normal = torch.bmm(normals_2d, normals_3d.transpose(1, 2)) # [B, HW, N]
        normal_mask = cos_normal < tau
        
        # Combinar máscaras (Si incumple distancia O normales -> -inf)
        m_geo = torch.zeros_like(attn_logits)
        m_geo[dist_mask | normal_mask] = -float('inf')
        
        # 4. Softmax enmascarado
        attn_weights = torch.softmax(attn_logits + m_geo, dim=-1) # [B, HW, N]
        
        # 5. Salida de Atención Cruzada
        out = torch.bmm(attn_weights, V) # [B, HW, C]
        return out

def compute_mnn_matches(feat_2d: torch.Tensor, feat_3d: torch.Tensor) -> torch.Tensor:
    # L2 normalizar para calcular similitud coseno por producto punto
    feat_2d_norm = nn.functional.normalize(feat_2d, p=2, dim=-1)
    feat_3d_norm = nn.functional.normalize(feat_3d, p=2, dim=-1)
    
    # Matriz de similitud coseno: [HW, N]
    sim = torch.mm(feat_2d_norm, feat_3d_norm.t())
    
    # Vecinos más cercanos de 2D a 3D
    nn_2d_to_3d = torch.argmax(sim, dim=1) # [HW]
    
    # Vecinos más cercanos de 3D a 2D
    nn_3d_to_2d = torch.argmax(sim, dim=0) # [N]
    
    # Condición MNN: i y j se eligen mutuamente
    indices_2d = torch.arange(feat_2d.size(0), device=feat_2d.device)
    mnn_mask = (nn_3d_to_2d[nn_2d_to_3d] == indices_2d)
    
    matches_2d = indices_2d[mnn_mask]
    matches_3d = nn_2d_to_3d[mnn_mask]
    
    return torch.stack([matches_2d, matches_3d], dim=1) # [M, 2]
```
