import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

def configure_optimizers(
    model: nn.Module,
    epochs: int,
    steps_per_epoch: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Configure AdamW optimizer and CosineAnnealing with Warmup scheduler.
    """
    # 1. Congelamiento de Parámetros:
    # Identificar las capas correspondientes a DINOv2 y configurar explícitamente requires_grad = False
    # Asegurar que los parámetros de los Visual Adapters, decodificador FPN y DGCNN mantengan requires_grad = True
    for name, param in model.named_parameters():
        if "dinov2" in name or "backbone" in name:
            if "adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Check that there are trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in the model.")

    # 2. Configuración del Optimizador (AdamW)
    # Tasa de aprendizaje base (eta_base): 1.0e-4
    # Decaimiento de pesos (weight decay): 1.0e-4
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

    # 3. Configuración del Scheduler
    eta_base = 1e-4
    eta_min = 1e-6
    T_warmup = 5 * steps_per_epoch
    T_max = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < 0:
            step = 0
        if step > T_max:
            step = T_max
            
        if step < T_warmup:
            eta_step = eta_min + (step / T_warmup) * (eta_base - eta_min)
        else:
            if T_max == T_warmup:
                eta_step = eta_min
            else:
                progress = (step - T_warmup) / (T_max - T_warmup)
                eta_step = eta_min + 0.5 * (eta_base - eta_min) * (1.0 + math.cos(math.pi * progress))
        # LambdaLR scales the base learning rate (1e-4) by this returned factor
        return eta_step / eta_base

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Perform a single training step.
    """
    # 1. Activar modo de entrenamiento en el modelo
    model.train()
    
    # Asegurar que las capas del backbone DINOv2 permanezcan congeladas (ej. eval mode para BatchNorm/Dropout)
    for name, module in model.named_modules():
        if ("backbone" in name or "dinov2" in name) and "adapter" not in name:
            module.eval()
            
    # Asegurar que los adaptadores estén explícitamente en modo entrenamiento
    for name, module in model.named_modules():
        if "adapter" in name:
            module.train()

    # 2. Transferir los tensores al dispositivo especificado
    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(device)
        else:
            batch_device[k] = v

    # 3. Reiniciar gradientes del optimizador
    optimizer.zero_grad()

    # 4. Paso forward
    feat_2d, feat_3d = model(batch_device['image'], batch_device['point_cloud'])
    
    # Asegurar que ambos descriptores salgan normalizados L2 en la dimensión C
    feat_2d = F.normalize(feat_2d, p=2, dim=-1)
    feat_3d = F.normalize(feat_3d, p=2, dim=-1)

    # 5. Evaluar función de pérdida
    loss = loss_fn(feat_2d, feat_3d, batch_device['gt_mask'])

    # 6. Paso backward
    loss.backward()

    # 7. Actualizar parámetros
    optimizer.step()

    # 8. Actualizar tasa de aprendizaje
    scheduler.step()

    # 9. Retornar estadísticas
    # Utilizar .item() para evitar retener el grafo de cómputo y fugas de memoria
    return {
        'loss': float(loss.item()),
        'lr': float(scheduler.get_last_lr()[0])
    }
