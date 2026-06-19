# 04. Función de Pérdida y Entrenamiento

El entrenamiento de la arquitectura GSCA se formula como un problema de **aprendizaje de métricas (Metric Learning)** en un espacio latente compartido $\mathcal{Z}$.

---

## 1. El Desafío de la Redundancia Estructural y Limitación de Triplet Loss

Los afloramientos geológicos exhiben patrones sedimentarios cíclicos y redes de fracturamiento altamente repetitivas. Esto genera texturas visuales y geometrías casi idénticas en múltiples ubicaciones físicas del macizo rocoso.

### La Limitación de Triplet Loss
Tradicionalmente, el matching inter-dominio se entrena usando **Triplet Loss**. Sin embargo, esta función aplica un gradiente uniforme a todos los pares dentro de un margen rígido:
1. Trata por igual a un par negativo fácil que a uno extremadamente difícil o ambiguo.
2. En entornos con alta repetición fractal, esto genera un sobreajuste a texturas estocásticas o provoca el colapso del entrenamiento (gradientes que se anulan o explotan).

---

## 2. Circle Loss con Self-paced Weighting

Para resolver la redundancia estructural, GSCA utiliza **Circle Loss**. A diferencia del margen rígido de Triplet Loss, Circle Loss proporciona una frontera de decisión circular flexible y pondera dinámicamente cada par según su dificultad.

### Formulación Matemática
La pérdida se calcula sobre un conjunto de $m$ similitudes de pares positivos y $n$ similitudes de pares negativos:

$$\mathcal{L}_{circle} = \log \left( 1 + \sum_{j=1}^{n} e^{\gamma \alpha_n^j (s_n^j - \Delta_n)} \sum_{i=1}^{m} e^{-\gamma \alpha_p^i (s_p^i - \Delta_p)} \right)$$

Donde:
* $s_p^i$ es la similitud coseno del $i$-ésimo par positivo (puntos correspondientes en 2D y 3D).
* $s_n^j$ es la similitud coseno del $j$-ésimo par negativo (puntos no correspondientes).
* $\gamma$ es un factor de escala hiperparamétrico.
* $\Delta_p$ y $\Delta_n$ son los márgenes de holgura para positivos y negativos.

### Mecanismo de Ponderación Dinámica (Self-paced Weighting)
Los factores de ponderación $\alpha_p^i$ y $\alpha_n^j$ regulan la influencia de cada par en el gradiente:

$$\alpha_p^i = [O_p - s_p^i]_+$$
$$\alpha_n^j = [s_n^j - O_n]_+$$

Donde:
* $[\cdot]_+ = \max(\cdot, 0)$ es la función ReLU.
* $O_p$ y $O_n$ son los objetivos de convergencia óptimos. Típicamente, si definimos un margen $\Delta$, se parametrizan como:
  $$O_p = 1 + \Delta \quad \text{y} \quad O_n = -\Delta$$
  y los márgenes se configuran como:
  $$\Delta_p = 1 - \Delta \quad \text{y} \quad \Delta_n = \Delta$$

### ¿Cómo actúa como currículo de aprendizaje?
* **Pares Consolidados**: Si un par positivo ya tiene una similitud muy alta ($s_p^i \approx O_p$), entonces $\alpha_p^i \to 0$. La presión de optimización disminuye, evitando que la red altere descriptores ya alineados.
* **Pares Ambiguos (Casos Difíciles)**: Si un par negativo tiene una similitud alta (confusión), $\alpha_n^j$ aumenta, amplificando el gradiente para empujar y separar estos descriptores.
* Esto evita el colapso del modelo ante texturas visuales idénticas en superficies rocosas distintas y fuerza al alineamiento a anclarse en discontinuidades físicas estables.

---

## 3. Estrategia de Optimización End-to-End

El entrenamiento de GSCA se realiza de extremo a extremo:
1. **Flujo de gradientes**: Los gradientes de $\mathcal{L}_{circle}$ se propagan de vuelta a través de:
   * Los **Visual Adapters** (PEFT) y la red de decodificación FPN en la **Rama Visual (2D)**.
   * Los parámetros del extractor DGCNN en la **Rama Geométrica (3D)**.
2. **Pesos Congelados**: El backbone de DINOv2 se mantiene congelado para preservar las características semánticas de propósito general.
3. **Hiperparámetros de entrenamiento recomendados**:
   * **Optimizar**: AdamW con weight decay de $10^{-4}$.
   * **Scheduler**: Cosine Annealing con warm-up de 5 épocas.
   * **Factor de Escala $\gamma$**: Configurado en un rango entre 64 y 128.
   * **Margen $\Delta$**: Configurado típicamente en 0.25 (lo que define $O_p=1.25$, $O_n=-0.25$, $\Delta_p=0.75$, $\Delta_n=0.25$).

---

## 4. Implementación del Módulo de Pérdida en PyTorch

```python
import torch
import torch.nn as nn

class CircleLoss2D3D(nn.Module):
    def __init__(self, gamma: float = 80.0, margin: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        
        # Objetivos de optimización óptimos
        self.O_p = 1.0 + margin
        self.O_n = -margin
        
        # Márgenes para la pérdida
        self.Delta_p = 1.0 - margin
        self.Delta_n = margin

    def forward(self, sim_matrix: torch.Tensor, ground_truth_mask: torch.Tensor) -> torch.Tensor:
        # sim_matrix: [N_2D, N_3D] Matriz de similitud coseno entre descriptores
        # ground_truth_mask: [N_2D, N_3D] Máscara booleana (True si hay correspondencia real)
        
        # Extraer similitudes de pares positivos y negativos
        pos_pair = sim_matrix[ground_truth_mask]
        neg_pair = sim_matrix[~ground_truth_mask]
        
        if len(pos_pair) == 0 or len(neg_pair) == 0:
            return torch.tensor(0.0, requires_grad=True, device=sim_matrix.device)
            
        # 1. Calcular pesos dinámicos alpha
        alpha_p = torch.clamp(self.O_p - pos_pair.detach(), min=0.0)
        alpha_n = torch.clamp(neg_pair.detach() - self.O_n, min=0.0)
        
        # 2. Aplicar márgenes y escalar con gamma
        logit_p = -self.gamma * alpha_p * (pos_pair - self.Delta_p)
        logit_n = self.gamma * alpha_n * (neg_pair - self.Delta_n)
        
        # 3. Suma exponencial para la pérdida tipo Log-Sum-Exp
        loss_p = torch.logsumexp(logit_p, dim=0) if len(logit_p) > 0 else torch.tensor(0.0)
        loss_n = torch.logsumexp(logit_n, dim=0) if len(logit_n) > 0 else torch.tensor(0.0)
        
        # L_circle = log( 1 + sum(exp(logit_n)) * sum(exp(logit_p)) )
        # Usando la propiedad log(1 + exp(A) * exp(B)) = log(1 + exp(A + B))
        loss = torch.log1p(torch.exp(loss_p + loss_n))
        
        return loss
```
