from typing import Dict
import logging
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def write_metrics(metrics: Dict[str, float], epoch: int, prefix: str, writer: SummaryWriter):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

def save_states(save_path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None, ema = None, **kwargs):
    save_dict = {'model_state_dict': model.state_dict(), 'model_structure': model}
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
        if getattr(scheduler, 'after_scheduler', None) is not None:
            save_dict['after_scheduler_state_dict'] = scheduler.after_scheduler.state_dict()
    if ema is not None:
        save_dict['ema_state_dict'] = ema.state_dict()
    
    for key, value in kwargs.items():
        save_dict[key] = value
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    
def load_states(load_path: str, device: torch.device, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None, ema = None, **kwargs):
    checkpoint = torch.load(load_path, map_location=device)
    
    if 'state_dict' not in checkpoint and 'model_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if getattr(scheduler, 'after_scheduler', None) is not None:
            scheduler.after_scheduler.load_state_dict(checkpoint['after_scheduler_state_dict'])
    if ema is not None:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        
    for key, value in kwargs.items():
        key = '{}_state_dict'.format(key)
        if key not in checkpoint:
            logger.warning(f'Key {key} not found in checkpoint')
        kwargs[key].load_state_dict(checkpoint[key])
        
def load_model(load_path: str, network_name: str='ResNet_3D_CPM'):
    checkpoint = torch.load(load_path)
    # Build model
    if 'model_structure' in checkpoint:
        model = checkpoint['model_structure']
    else:
        if network_name == 'ResNet_3D_CPM':
            from networks.ResNet_3D_CPM import Resnet18
        elif network_name == 'ResNet_3D_CPM_stride2':
            from networks.ResNet_3D_CPM_stride2 import Resnet18
        elif network_name == 'ResNet_3D_CPM_multiHead':
            from networks.ResNet_3D_CPM_multiHead import Resnet18
        elif network_name == 'ResNet_3D_CPM_multiHead_sride2':
            from networks.ResNet_3D_CPM_multiHead_sride2 import Resnet18
        elif network_name == 'ResNet_3D_CPM_SCconv':
            from networks.ResNet_3D_CPM_SCconv import Resnet18
        elif network_name == 'ResNet_3D_CPM_dynamicK':
            from networks.ResNet_3D_CPM_dynamicK import Resnet18
        else:
            print('network_name can\'t find')
            exit()
        model = Resnet18()
        
    # Load state dict
    if 'state_dict' not in checkpoint and 'model_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_memory_format(memory_format: str):
    if memory_format == 'channels_last':
        return torch.channels_last_3d
    else:
        return None