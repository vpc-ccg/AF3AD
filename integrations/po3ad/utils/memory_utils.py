import logging
import subprocess
from typing import Dict, Optional

import psutil
import torch


def get_memory_info() -> Dict[str, float]:
    """
    Get current system and GPU memory usage.
    
    Returns:
        dict: Memory information including:
            - system_ram_used_gb: System RAM used in GB
            - system_ram_percent: System RAM usage percentage
            - system_ram_available_gb: Available system RAM in GB
            - gpu_X_allocated_mb: GPU X memory allocated in MB (for each GPU)
            - gpu_X_reserved_mb: GPU X memory reserved in MB (for each GPU)
            - gpu_X_total_mb: GPU X total memory in MB (for each GPU)
            - gpu_X_percent: GPU X memory usage percentage (for each GPU)
    """
    memory_info = {}
    
    # System memory
    try:
        vm = psutil.virtual_memory()
        memory_info['system_ram_used_gb'] = vm.used / (1024 ** 3)
        memory_info['system_ram_percent'] = vm.percent
        memory_info['system_ram_available_gb'] = vm.available / (1024 ** 3)
        memory_info['system_ram_total_gb'] = vm.total / (1024 ** 3)
    except Exception as e:
        memory_info['system_ram_error'] = str(e)
    
    # GPU memory (PyTorch) - only check if CUDA is available and initialized
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            device_count = torch.cuda.device_count()
            for gpu_id in range(device_count):
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 2)  # MB
                
                # Get max memory for percentage calculation
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory = props.total_memory / (1024 ** 2)  # MB
                
                memory_info[f'gpu_{gpu_id}_allocated_mb'] = allocated
                memory_info[f'gpu_{gpu_id}_reserved_mb'] = reserved
                memory_info[f'gpu_{gpu_id}_total_mb'] = total_memory
                # Use reserved memory for percentage as it includes fragmentation overhead
                memory_info[f'gpu_{gpu_id}_percent'] = (reserved / total_memory * 100) if total_memory > 0 else 0
    except (RuntimeError, AttributeError) as e:
        memory_info['gpu_error'] = str(e)
    
    return memory_info


def get_nvidia_smi_info() -> Optional[str]:
    """
    Get GPU memory information from nvidia-smi command.
    
    Returns:
        str: Output from nvidia-smi command, or None if command fails
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def report_memory_usage(logger: logging.Logger, epoch: int, prefix: str = ""):
    """
    Report current memory usage to logger.
    
    Args:
        logger: Logger instance to write to
        epoch: Current epoch number
        prefix: Optional prefix for log messages
    """
    memory_info = get_memory_info()
    
    # Build log message
    msg_parts = []
    
    # System memory
    if 'system_ram_used_gb' in memory_info:
        msg_parts.append(
            f"RAM: {memory_info['system_ram_used_gb']:.2f}/{memory_info['system_ram_total_gb']:.2f} GB "
            f"({memory_info['system_ram_percent']:.1f}%)"
        )
    
    if msg_parts:
        prefix_str = f"{prefix} " if prefix else ""
        logger.info(f"{prefix_str}E{epoch:03d} Memory | {' | '.join(msg_parts)}")
    
    # Also log nvidia-smi output if available
    nvidia_info = get_nvidia_smi_info()
    if nvidia_info:
        logger.info(f"{prefix_str}E{epoch:03d} nvidia-smi:")
        for line in nvidia_info.split('\n'):
            if line.strip():
                # Parse CSV: index,name,memory.used,memory.total,utilization.gpu
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_idx, gpu_name, mem_used, mem_total, gpu_util = parts[:5]
                    logger.info(f"  GPU{gpu_idx} ({gpu_name}): {mem_used}/{mem_total} MB, Util: {gpu_util}%")


def clear_cuda_cache():
    """Clear PyTorch CUDA cache to free unused memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()