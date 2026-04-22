import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed) # 해시 함수 난수성 고정
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # 최신 GPU 결정론적 연산 강제

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def seed_worker(worker_id: int) -> None:
    # PyTorch DataLoader이 멀티 프로세싱으로 동작할 때 워커들의 시드 고정
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    # 데이터 로딩 순서 보호
    g = torch.Generator()
    g.manual_seed(seed)
    return g