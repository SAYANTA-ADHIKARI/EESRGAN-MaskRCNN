import torch
import subprocess

def get_gpu_memory_map():
    """
    Get the current GPU usage.
    Returns
    -------
    memory_map: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
            '--format=csv,nounits,noheader'
        ], encoding='utf-8'
    )
    # Lines look like '3241, 4096, 56'
    gpu_info = result.strip().split('\n')
    memory_map = {}
    for i, info in enumerate(gpu_info):
        used_memory, total_memory, utilization = map(int, info.split(','))
        memory_map[i] = {
            'used_memory': used_memory,
            'total_memory': total_memory,
            'utilization': utilization
        }
    return memory_map

def print_gpu_utilization():
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("No GPU detected.")
        return

    # Get memory usage and utilization for each GPU
    gpu_memory_map = get_gpu_memory_map()

    for gpu_id in range(num_gpus):
        gpu_info = gpu_memory_map[gpu_id]
        print(f"GPU {gpu_id}:")
        print(f"  Memory Usage: {gpu_info['used_memory']} MiB / {gpu_info['total_memory']} MiB")
        print(f"  Utilization: {gpu_info['utilization']}%")
        print()

# Run the function to print GPU utilization
print_gpu_utilization()