"""DDP setup helpers for the KD trainer."""
import os
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup_ddp(backend: str = "nccl", master_addr: str = "127.0.0.1", master_port: str = "29500"):
    """torchrun이 주입한 env vars(LOCAL_RANK, RANK, WORLD_SIZE)로 DDP 초기화."""
    if "RANK" not in os.environ:
        # 단일 GPU 모드 (디버깅)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if int(os.environ["WORLD_SIZE"]) > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return local_rank, get_rank(), get_world_size()


def cleanup_ddp():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def barrier():
    if is_dist():
        dist.barrier()
