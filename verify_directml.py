import torch
print(f"PyTorch version: {torch.__version__}")

# This script is designed to test the DirectML backend for PyTorch.
# At the time of writing, DirectML support in PyTorch is experimental.
# The script remains a useful test to see if the DirectML backend can be imported
# and if it can create tensors on a DirectML device.

try:
    import torch_directml
    print("Successfully imported 'torch_directml' module.")
except ImportError as e:
    print(f"Failed to import 'torch_directml' module: {e}")
except Exception as e:
    print(f"An unexpected error occurred during 'torch_directml' import: {e}")


dml_device_name = "privateuseone:0"
device_str = "cpu"

try:
    print(f"Attempting to create tensor on device: {dml_device_name}...")
    
    # This is the main test
    test_tensor = torch.tensor([1.0, 2.0]).to(dml_device_name)
    
    if test_tensor.device.type == 'privateuseone':
        print(f"Successfully created tensor on DirectML device: {test_tensor.device}")
        device_str = dml_device_name
    else:
        print(f"Tensor created, but on device: {test_tensor.device.type}. Expected 'privateuseone'.")
        print("DirectML does not seem to be fully functional.")

except Exception as e:
    print(f"Failed to use DirectML device ({dml_device_name}). Error: {e}")
    print("PyTorch will use CPU.")

final_device = torch.device(device_str)
print(f"PyTorch is using device: {final_device}")

if final_device.type == 'privateuseone':
    print("DirectML backend appears to be working!")
else:
    print("DirectML backend is NOT working.")