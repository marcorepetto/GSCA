# 01. Rama Visual (2D): Descriptores Semánticos Densos con DINOv2 y Adaptores

La rama visual (2D) es responsable de mapear imágenes RGB del afloramiento geológico a descriptores semánticos densos que sean robustos ante la variabilidad lumínica extrema.

---

## 1. Selección del Backbone: DINOv2

La arquitectura adopta **DINOv2** como la columna vertebral para extraer características visuales, en lugar de enfoques tradicionales basados en intensidades locales (como SuperPoint o SIFT) o alineamientos texto-imagen (como CLIP).

### Limitaciones de SuperPoint en Afloramientos
SuperPoint confía en gradientes de intensidad de brillo local para detectar esquinas y bordes. En macizos rocosos, la reflectancia está gobernada por la **BRDF (Bidirectional Reflectance Distribution Function)**. La BRDF cambia drásticamente con la posición del sol y la rugosidad a micro-escala, provocando sombras transitorias que la red confunde con discontinuidades físicas (fracturas o fallas).

### Ventajas de DINOv2
DINOv2 se pre-entrena mediante:
* **Destilación a nivel de imagen**: Permite capturar la semántica global.
* **iBOT (Masked Image Modeling)**: Fuerza al transformador visual (ViT) a reconstruir parches enmascarados desde su contexto local, induciendo una comprensión profunda de las estructuras granulares.
* **Resolución densa**: Cada token de parche funciona como un descriptor con alta coherencia espacial y sensibilidad local.

---

## 2. Adaptación de Dominio: Visual Adapters (PEFT)

A pesar de las capacidades de DINOv2, los modelos fundacionales tienen un sesgo hacia bajas frecuencias espaciales (formas, siluetas globales). En geología, la información clave está en las altas frecuencias (micro-rugosidad de fallas, anisotropía de fracturas, estratos delgados).

Para adaptar el modelo sin incurrir en el **olvido catastrófico** de su representación global, se utilizan **Visual Adapters** mediante técnicas de Ajuste Fino Eficiente en Parámetros (PEFT).

### Formulación del Adaptador
Se inyectan pequeños cuellos de botella (bottleneck layers) en paralelo a las capas Feed-Forward (MLP) de cada bloque del Transformer de DINOv2, manteniendo el resto de la red congelada:

$$\text{Adapter}(\mathbf{x}) = \sigma(\mathbf{x} \mathbf{W}_{down}) \mathbf{W}_{up}$$

Donde:
* $\mathbf{x} \in \mathbb{R}^{d}$ es la representación intermedia del token.
* $\mathbf{W}_{down} \in \mathbb{R}^{d \times r}$ es una proyección de reducción de dimensionalidad con rango intrínseco $r \ll d$.
* $\sigma(\cdot)$ es una función de activación no lineal (por ejemplo, GELU o ReLU).
* $\mathbf{W}_{up} \in \mathbb{R}^{r \times d}$ es la proyección de vuelta a la dimensión del Transformer.

Este diseño asegura que los pesos pre-entrenados del Transformer permanezcan inalterados mientras los adaptadores sintonizan los descriptores hacia el dominio estructural geológico.

---

## 3. Feature Pyramid Network (FPN) y Resolución Densa

DINOv2 opera dividiendo la imagen en parches (normalmente de $14 \times 14$ píxeles), lo que resulta en una resolución de descriptores inferior a la de la imagen original.

Para recuperar la resolución espacial densa a nivel de píxel (necesaria para el matching preciso), se acopla una **Feature Pyramid Network (FPN)** ligera o un decodificador tipo **Dense Prediction Transformer (DPT)** sobre múltiples capas intermedias del ViT. Esto permite combinar mapas de características de alta resolución y bajo nivel con mapas semánticos profundos de alta dimensionalidad.

---

## 4. Guía de Implementación en PyTorch

A continuación se presenta un bosquejo en PyTorch de cómo implementar el módulo del adaptador en paralelo al MLP de un bloque de DINOv2:

```python
import torch
import torch.nn as nn

class VisualAdapter(nn.Module):
    def __init__(self, embed_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim)
        )
        # Inicialización de pesos cerca de cero para mantener estabilidad al inicio
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Tokens, EmbedDim]
        return self.adapter(x)

class AdaptedTransformerBlock(nn.Module):
    def __init__(self, original_block: nn.Module, bottleneck_dim: int = 64):
        super().__init__()
        # Se hereda o envuelve el bloque original de DINOv2/ViT
        self.original_block = original_block
        embed_dim = original_block.mlp.fc1.in_features
        
        # Congelar los pesos del bloque original
        for param in self.original_block.parameters():
            param.requires_grad = False
            
        # Crear e inyectar el adaptador entrenable en paralelo al MLP
        self.adapter = VisualAdapter(embed_dim=embed_dim, bottleneck_dim=bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Se computa el paso por la atención (que permanece congelada)
        x_attn = x + self.original_block.drop_path1(
            self.original_block.ls1(self.original_block.attn(self.original_block.norm1(x)))
        )
        
        # El bloque MLP original corre en paralelo al adaptador
        mlp_out = self.original_block.drop_path2(
            self.original_block.ls2(self.original_block.mlp(self.original_block.norm2(x_attn)))
        )
        adapter_out = self.adapter(self.original_block.norm2(x_attn))
        
        # Salida combinada
        return x_attn + mlp_out + adapter_out
```

### Decodificador FPN Ligero
Para proyectar los tokens del ViT ($f_{patch} \in \mathbb{R}^{H_p \times W_p \times C}$) a la resolución del píxel original, se extraen las activaciones de los bloques $l \in \{4, 8, 12, 16\}$ (para ViT-Large/Base) y se procesan con convoluciones bilineales y concatenaciones laterales para obtener un mapa denso $\mathbf{f}_i^{2D} \in \mathbb{R}^{H \times W \times C_{final}}$.
