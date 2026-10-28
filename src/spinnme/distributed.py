"""Rank identity before Lightning initializes the process group."""
import os

from pytorch_lightning.utilities import rank_zero_only


def is_primary_process():
    # Lightning's subprocess launcher sets LOCAL_RANK but can inherit
    # SLURM_PROCID=0 (or RANK=0) from its parent. Any nonzero rank is a reader.
    keys = ('RANK', 'LOCAL_RANK', 'NODE_RANK', 'GROUP_RANK', 'SLURM_PROCID', 'SLURM_NODEID', 'JSM_NAMESPACE_RANK')
    ranks = [int(os.environ[key]) for key in keys if key in os.environ]
    return all(rank == 0 for rank in ranks) if ranks else rank_zero_only.rank == 0
