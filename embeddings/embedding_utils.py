import torch


def print_tensor_info(tensor):

    print("Shape:", tensor.shape)

    print("Dtype:", tensor.dtype)

    print("Device:", tensor.device)