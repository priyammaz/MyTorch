"""
Using standard safetensors to save and load weights
"""
from safetensors.numpy import save_file, load_file

def save(state_dict, filepath):
    save_file(state_dict, filepath)

def load(filepath):
    return load_file(filepath)
