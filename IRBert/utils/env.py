import os
import shutil
import socket

import torch
import torch.distributed as dist

__all__ = ['init_directory', 'clean_directory', 'setup_distributed_env', 
        'cleanup_distributed_env', 'tcp_port_in_use']

def init_directory(path):
    assert isinstance(path, str)

    if not os.path.isdir(path):
        os.makedirs(path)

    return path

def clean_directory(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {0}. Reason: {1}'.format(f, e))

    return None

def setup_distributed_env(rank, world_size, init_method = 'env://', backend = 'gloo'):
    # CPU use 'gloo', CUDA GPU use 'nccl'

    assert isinstance(rank, int)
    assert isinstance(world_size, int)
    assert isinstance(init_method, str)
    assert isinstance(backend, str)
    assert backend in ['gloo', 'nccl']

    dist.init_process_group(backend = backend,
                            init_method = init_method,
                            world_size = world_size,
                            rank = rank)

    return None

def cleanup_distributed_env():
    dist.destroy_process_group()
    return None

def tcp_port_in_use(host, port):
    if not isinstance(host, str):
        raise TypeError('Argument: host must be a Python string object.')

    if not isinstance(port, int):
        raise TypeError('Argument: port must be a Python int object.')

    if port < 0:
        raise ValueError('Argument: port must larger than zero.')

    in_use = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use = (s.connect_ex((host, port)) == 0)

    return in_use


