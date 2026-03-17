import gc
import torch

def clear_gpu_memory():
    # Collect garbage
    gc.collect()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    # Optionally, reset all devices
    torch.cuda.reset_peak_memory_stats()
    
    # Print current memory usage
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Call this between runs or after intensive computations ahhh okay
clear_gpu_memory()