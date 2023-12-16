import os
from collections import deque
from contextlib import contextmanager

import torch
import torch.distributed as dist
from smdistributed.modelparallel.backend.config import ModelParallelConfig
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.random import RngManager

from .module_manager import ModuleManager

cfg = None


def create_process_groups():
    global tp_process_group
    tp_process_group = dist.new_group(list(range(cfg.tensor_parallel_degree)))


def set_cfg(config):
    global cfg
    cfg = ModelParallelConfig(config)
    create_process_groups()


class NCCLThrottler:
    """
    NCCL holds on to the input and output buffers for a collective as long as it is in progress.
    Because of this, if there are too many NCCL calls queued up at the same time, these buffers
    end up take up too mcuh memory. This throttler ensures that there are at most self.limit
    NCCL calls in progress at a time. When at the limit, it waits for an existing call to finish
    before scheduling a new one.
    """

    def __init__(self):
        self.call_queue = deque()
        self.event_stack = []
        self.limit = None
        self.enabled = False

    def set_throttle_limit(self):
        self.limit = int(os.environ.get("SMP_NCCL_THROTTLE_LIMIT", 8))

        # disable throttling for non-positive limits
        self.enabled = self.limit > 0
        if rank() == 0 and self.enabled:
            logger.info(f"Using NCCL throttle limit of {self.limit}.")

    @contextmanager
    def throttle(self):
        """Throttle the NCCL call. Any NCCL calls placed inside will be deferred until the number of ongoing NCCL
        calls (that were also launched within this context) drops below the limit."""

        if self.enabled:
            torch.cuda.set_device(local_rank())
            stream = torch.cuda.current_stream()

            if len(self.call_queue) == self.limit:
                self.call_queue[0].synchronize()
                complete_event = self.call_queue.popleft()
                self.event_stack.append(complete_event)

        yield

        if self.enabled:
            if len(self.event_stack) == 0:
                # create new event if there are no free ones
                event = torch.cuda.Event()
                self.event_stack.append(event)

            event = self.event_stack.pop()
            stream.record_event(event)
            self.call_queue.append(event)


rng_manager = RngManager()
module_manager = ModuleManager()
nccl_throttler = NCCLThrottler()
tp_process_group = None
logger = get_logger()


def local_rank():
    return dist.get_rank() % torch.cuda.device_count()


def pp_rank():
    return 0


def pp_size():
    return 1


def rank():
    return dist.get_rank()


def rdp_rank():
    return 0


def rdp_size():
    return dist.get_world_size()


def tp_rank():
    return dist.get_rank() % cfg.tensor_parallel_degree


def tp_size():
    return cfg.tensor_parallel_degree


def get_tp_process_group():
    return tp_process_group
