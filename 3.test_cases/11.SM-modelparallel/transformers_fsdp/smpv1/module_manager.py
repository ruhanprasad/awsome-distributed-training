# Standard Library
import os
from collections import OrderedDict, defaultdict, deque, namedtuple
from contextlib import contextmanager
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

# Third Party
import torch
import torch.nn as nn

# First Party
from smdistributed.modelparallel.backend.core import get_logger
from torch.utils.hooks import RemovableHandle

logger = get_logger()
SMPUnsupportedError = RuntimeError
CheckpointingConfigError = RuntimeError
DistributedModelNotWrappedError = RuntimeError
DistributedModelWrappedError = RuntimeError
InvalidPartitionIDError = RuntimeError
MissingModuleError = RuntimeError
MissingOutputForModuleError = RuntimeError
MissingParentModuleError = RuntimeError
PipelineParallelBWDError = RuntimeError
SMPCheckpointError = RuntimeError
SMPRuntimeError = RuntimeError
StepFunctionCalledError = RuntimeError
UnassignedPartitionError = RuntimeError
UnsupportedCommunicationVolumeUnitError = RuntimeError


def rmsg(msg):
    return f"[] {msg}"


TensorModuleInfo = namedtuple(
    "TensorModuleInfo", "module_name count tensor_idx is_forward end_index"
)


class TraceResults(NamedTuple):
    mod_execution_order: List[nn.Module]
    traced_input_sizes: Dict[nn.Module, int]
    traced_output_sizes: Dict[nn.Module, int]
    mod_execution_times: Dict[nn.Module, float]
    mod_memory_usage: Dict[nn.Module, float]


class PartitioningAndTraceResults(NamedTuple):
    mod_partitions: Dict[str, int]
    mod_execution_order: List[str]
    traced_input_sizes: Dict[str, int]
    traced_output_sizes: Dict[str, int]
    mod_execution_times: Dict[str, float]
    mod_memory_usage: Dict[str, float]

    def __repr__(self):
        return "<PartTraceResults>"


class ModuleManager:
    def __init__(self):
        self.cfg = None
        self.reset()

    def set_config(self, config):
        self.cfg = config

    def reset(self):
        """
        This helps reset all state so we can run on a different model after reset
        """
        # maintain manual partition assignment
        self._module_partitions: Dict[nn.Module, int] = {}
        self._cur_partition: Optional[int] = None

        # set of activation_checkpoint configs of modules
        self._activation_checkpoint_modules_config = {}

        # flag representing whether tensor parallelism is currently enabled through the manual API.
        # tensor parallelism will be activated on a best-effort basis whenever this is True.
        self._tensor_parallelism_enabled = False

        # set of modules for which we will apply tensor parallelism
        self._tensor_parallelism_modules = set()

        # mapping from module to dict containing optional config for distribution
        self._tensor_parallelism_config = {}

        # the configuration to be used for the tensor-parallel modules marked in the current context
        self._current_tp_config = {}

        # set of modules that are tensor-parallelized
        self._distributed_modules = set()

        # set of parameters that operate on the batch scaled by tp_size()
        self._scaled_batch_parameters = set()
        self._scaled_batch_buffers = set()

        # set of parameters which are present only on one tp_rank() (tp_rank() == 0)
        # and set to None on others
        self._one_rank_parameters = set()
        self._one_rank_buffers = set()

        # set of parameters that are distributed across tp_ranks. typically will
        # be the same as self._scaled_batch_parameters, but does not have to, depending
        # on how the parameter_creation_scope / initialize_with_input_partition contexts
        # are used in DistributedModule implementation
        self._distributed_parameters = {}
        self._distributed_buffers = {}

        # collect information from tracing
        self._traced_input_sizes: Dict[nn.Module, int] = {}
        self._traced_output_sizes: Dict[nn.Module, int] = {}
        self._module_execution_order: List[nn.Module] = []
        self._module_execution_times: Dict[nn.Module, float] = {}

        # CUDA events that mark the start and end of each module execution during tracing
        self._mod_execution_cuda_events: Dict[nn.Module, Tuple] = {}

        self._module_memory_usage: Dict[nn.Module, float] = {}

        # used for serialization and deserialization as the key to above dicts is a module
        # object local to a process. These dicts help convert them to a string so information
        # can be sent to other processes
        self._module_to_name: Dict[nn.Module, str] = {}
        self._name_to_module: Dict[str, nn.Module] = {}

        # mapping from child module names to parent module names
        self._parent_map: Dict[str, List[str]] = defaultdict(list)

        # sum of _to_recv counts of all children of a module (does not include the module itself)
        # (keyed by (microbatch, module_name))
        self._pending_bwd_counts: Dict[Tuple, int] = defaultdict(lambda: 0)

        # used to identify parent module
        # this is sent and received across ranks for each request
        self._module_execution_stack: List[nn.Module] = []

        # dict of dicts, with first key as microbatch, second key as module
        # and value as a deque data structure. This will be used a stack
        # to record outputs during forward for a microbatch and a module.
        # In the backward pass, outputs will be popped from stack and backward
        # called on them
        self._module_outputs: Dict[int, Dict[Tuple, Deque[Tuple[torch.Tensor]]]] = defaultdict(
            lambda: defaultdict(deque)
        )

        # records outputs and out_grads for a mb, parent_module and module
        # this data structure is used to wait on backward requests from child
        # and club multiple backward requests into a single request
        self._bwd_tensors: Dict[int, Dict[Tuple, Deque[Tuple[torch.Tensor]]]] = defaultdict(
            lambda: defaultdict(deque)
        )

        # stores the number of real responses yet to receive for backward requests. this will be incremented
        # when we have an additional backward request that we haven't received response for
        # when response is received (real) this is decremented
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_recv_real: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # stores the number of dummy repsonses yet to receive for backward requests. this will be incremented
        # when we have an additional backward request that we haven't received response for
        # when response is received (dummy) this is decremented
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_recv_dummy: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # stores the number of  repsonses yet to receive for backward requests for sequential modules. this will be incremented
        # when we have an additional backward request that we haven't received response for.
        # When response is received this is decremented
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_recv_sequential: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # stores the dummy backward request to be sent to a parent on a different rank
        # even if the inputs coming from parent module doesnt require grads, we need
        # to send dummy backward request to let the parent module know that backward execution
        # is complete for the child module. This maintains a count of num dummies to send.
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_send_dummy: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # flag to indicate if we need to send back dummy backward request to parent or not
        # dict structure is microbatch -> {position -> {parent_module_name -> {module_name -> bool}}}
        self._to_send_count: Dict[int, Dict[int : Dict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        )

        # mapping of node to bwd_counts for (mb, position, (mod, parent_mod))
        # dict structure is microbatch -> {position -> {(parent_module_name, module_name) -> {node -> int}}}
        self._smpinput_bwd_counts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )

        # records SMPParentRecv backward nodes for a microbatch
        # this data structure is used to do keep track of SMPParentRecv nodes, so that
        # it can be used to check for unsupported scenarios at the end of module
        # execution
        self._smpparents: Dict[int, Dict[str, Deque[Any]]] = defaultdict(
            lambda: defaultdict(lambda: set())
        )

        # records SMPParentRecv backward nodes for a microbatch
        # this data structure is used to do keep track of SMPInput nodes, so that
        # it can be used to check for unsupported scenarios at the end of module
        # execution
        self._smpinputs: Dict[int, Dict[str, Deque[Any]]] = defaultdict(
            lambda: defaultdict(lambda: set())
        )

        # records additional parameters which were passed in the forward
        # pass of the module
        self._additional_params: Dict[nn.Module, set] = defaultdict(set)

        # whether tracing is enabled
        self.measurement_enabled = False

        # The tensor parallel split shapes for a certain weight
        # key: weight tensor, value: list of split shapes for each tp rank
        # To record the unbalanced split and used during loading
        self.weight_split_shapes = {}

        # hooks to be called after the model partition
        self._post_partition_hooks = OrderedDict()

        # hooks to be called after the first execution of smp.step
        self._post_step_hooks = OrderedDict()

        # whether the partition is loaded from a checkpoint
        self.partition_loaded = False

        # list of tuple (smp_to_hf, hf_to_smp) state_dict translate function
        self.translate_functions = []

    def add_scaled_batch_parameter(self, param):
        self._scaled_batch_parameters.add(param)

    def add_distributed_parameter(self, param, axis):
        self._distributed_parameters[param] = axis

    def is_scaled_batch_parameter(self, param):
        return param in self._scaled_batch_parameters

    def get_parameter_distribution_axis(self, param):
        """If not distributed returns None"""
        return self._distributed_parameters.get(param, None)

    def add_scaled_batch_buffer(self, buf):
        self._scaled_batch_buffers.add(buf)

    def add_distributed_buffer(self, buf, axis):
        self._distributed_buffers[buf] = axis

    def add_one_rank_parameter(self, param):
        self._one_rank_parameters.add(param)

    def add_one_rank_buffer(self, buf):
        self._one_rank_buffers.add(buf)

    def is_one_rank_parameter(self, param):
        return param in self._one_rank_parameters

    def is_one_rank_buffer(self, buf):
        return buf in self._one_rank_buffers

    def is_scaled_batch_buffer(self, buf):
        return buf in self._scaled_batch_buffers

    def get_buffer_distribution_axis(self, buf):
        """If not distributed returns None"""
        return self._distributed_buffers.get(buf, None)

    def update_fake_tensor(self, fake_to_materialized_param, fake_to_materialized_buffer):
        """
        When initialized with deferred_init, recorded tensors are fake ones. Replace them with the materalized real tensors.
        """
        for fake, real in fake_to_materialized_param.items():
            if fake in self._scaled_batch_parameters:
                self._scaled_batch_parameters.remove(fake)
                self._scaled_batch_parameters.add(real)
            if fake in self._one_rank_parameters:
                self._one_rank_parameters.remove(fake)
                self._one_rank_parameters.add(real)
            if fake in self._distributed_parameters:
                axis = self._distributed_parameters[fake]
                del self._distributed_parameters[fake]
                self._distributed_parameters[real] = axis

        for fake, real in fake_to_materialized_buffer.items():
            if fake in self._scaled_batch_buffers:
                self._scaled_batch_buffers.remove(fake)
                self._scaled_batch_buffers.add(real)
            if fake in self._one_rank_buffers:
                self._one_rank_buffers.remove(fake)
                self._one_rank_buffers.add(real)
            if fake in self._distributed_buffers:
                axis = self._distributed_buffers[fake]
                del self._distributed_buffers[fake]
                self._distributed_buffers[real] = axis

    def register_translate_function(self, translate_functions):
        self.translate_functions.append(translate_functions)

    def get_model_partition_info(self):
        from smdistributed.modelparallel.torch.state_mod import state

        if not state.model.partitioned:
            raise SMPUnsupportedError(
                "get_model_partition_info can only be called after the model partition."
            )
        model_partition = {}
        for name, module in self._name_to_module.items():
            model_partition[name] = self._module_partitions[module]
        return model_partition

    def register_post_step_hook(self, hook):
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model is not None and state.optimizer is not None and state.model.partitioned:
            hook(state.model, state.optimizer)
            return None

        handle = RemovableHandle(self._post_step_hooks)
        self._post_step_hooks[handle.id] = hook
        return handle

    def register_post_partition_hook(self, hook):
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model is not None and state.optimizer is not None and state.model.partitioned:
            hook(state.model, state.optimizer)
            return None
        handle = RemovableHandle(self._post_partition_hooks)
        self._post_partition_hooks[handle.id] = hook
        return handle

    def replace_module(self, old_module, new_module):
        """Replace the original module with the new one in the internal data structures"""
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model.partitioned:
            raise SMPUnsupportedError(
                "Module replacement can only happen before the model partition."
            )

        name = self._module_to_name[old_module]
        self._module_to_name[new_module] = name
        self._name_to_module[name] = new_module
        del self._module_to_name[old_module]

        self._module_partitions[new_module] = self._module_partitions[old_module]
        del self._module_partitions[old_module]

    def record_traversal_into_module(self, module: nn.Module):
        self._module_execution_stack.append(module)

    def find_boundary_ancestors(self, mb: int, module: nn.Module) -> Tuple[str]:
        """Finds boundary ancestors for the module.
        Boundary ancestors means that an ancestor whose executor is not the current
        rank and a child of the ancestor whose executor is the current rank
        """
        current_mod = module
        current_mod_name = self.get_module_name(module)
        if not self.is_executor(current_mod):
            raise SMPRuntimeError("the rank needs to be executor of the module")
        if self.is_main_module(current_mod) or not self.is_parent_executor(current_mod):
            # parent of main is None
            return self.get_parent_module(current_mod), current_mod
        while not self.is_main_module(current_mod) and self.is_parent_executor(current_mod):
            child_mod = current_mod
            current_mod = self.get_parent_module(current_mod)
        child_mod = current_mod
        current_mod = self.get_parent_module(current_mod)
        return current_mod, child_mod

    def output_stack_size(self, mb: int, module: nn.Module, parent_module: nn.Module) -> int:
        """Returns output_stack_size given a microbatch, module and parent module
        This output stack size for a parent_module, module and a microbatch should be same
        on the rank executing parent_module and the rank executing module at the end of forward.
        """
        module_name = self.get_module_name(module)
        parent_module_name = self.get_module_name(parent_module)
        if (
            mb in self._module_outputs
            and (parent_module_name, module_name) in self._module_outputs[mb]
        ):
            return len(self._module_outputs[mb][(parent_module_name, module_name)])
        return 0

    def get_parent_module(self, module: nn.Module) -> nn.Module:
        """
        Traverse from end of the module execution stack to identify the immediate parent of given module.
        We do this from the end because we might have gone into this module multiple times.
        Returns the parent module.
        """
        if self.is_main_module(module):
            return None
        module_name = self.get_module_name(module)
        # TODO: Requires optimization
        if len(self.execution_stack):
            if module_name not in self.execution_stack:
                return self._module_execution_stack[-1]
            else:
                index = self.execution_stack.index(module_name)
                return self._module_execution_stack[index - 1] if index > 0 else None
        raise MissingParentModuleError(self.execution_stack, self.get_module_name(module))

    def get_parameters(self, module, recurse=True):
        """
        Yields the parameters owned by the model as well as
        additional parameters passed in forward.
        """
        for param in module.parameters(recurse=recurse):
            yield param
        for param in self._additional_params[module]:
            yield param

    def is_main_module(self, module: nn.Module) -> bool:
        """
        Check if the module is main module.
        main module is a module with no parent. In other words,
        its the module on which smp.distribute_model was called
        """
        return self.get_module_name(module) == "main"

    def is_correct_parent(self, module, parent_module):
        """Checks if the parent_module is the the correct parent of module"""
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        if self._parent_map[module_name]:
            return parent_module_name in self._parent_map[module_name]
        else:
            return self.is_main_module(module)

    def get_immediate_ancestors(self, module):
        """Gets parent and grand parent of a module"""
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        if self.is_main_module(module):
            raise SMPRuntimeError("cannot get_immediate_ancestors for main module")
        stack_parent_module = self.get_parent_module(module)
        module_name = self.get_module_name(module)
        stack_parent_name = self.get_module_name(stack_parent_module)
        ancestors = self._get_ancestors(module_name, stack_parent_name)
        if len(ancestors) < 2:
            raise SMPRuntimeError("ancestors should at least hold this module and parent")
        if len(ancestors) == 2:
            if self._parent_map[ancestors[-1]]:
                grand_parent_name = self._parent_map[ancestors[-1]][-1]
            else:
                grand_parent_name = None
            grand_parent_module = (
                None if not grand_parent_name else self.get_module(grand_parent_name)
            )
            parent_module = self.get_module(ancestors[-1])
        else:
            parent_module = self.get_module(ancestors[1])
            grand_parent_module = self.get_module(ancestors[2])

        return grand_parent_module, parent_module

    def _get_ancestors(self, module_name, stack_parent_name):
        """For a valid module name and stack parent module name
        returns the path from module to stack_parent"""
        visited = set()
        queue = deque()
        queue.append([module_name])
        while queue:
            current_path = queue.popleft()
            last = current_path[-1]
            if not last:
                continue
            elif last == stack_parent_name:
                return current_path
            if not self._parent_map[last]:
                continue
            for parent in self._parent_map[last]:
                if parent in visited:
                    continue
                visited.add(parent)
                new_path = list(current_path)
                new_path.append(parent)
                queue.append(new_path)
        raise SMPRuntimeError(f"path not found between {module_name} and {stack_parent_name}")

    def finished_module_exec(self):
        self._module_execution_stack.pop()

    def clear_microbatch_state(self, mb):
        self._to_recv_real.pop(mb, None)
        self._to_recv_dummy.pop(mb, None)
        self._to_recv_sequential.pop(mb, None)

        self._to_send_dummy.pop(mb, None)
        self._to_send_count.pop(mb, None)

        self._module_outputs.pop(mb, None)
        # not clearing self._pending_bwd_counts here as it is not keyed by microbatch
        # this is just int and should have no memory issues,
        # it's cleared at the end of step
        self._smpinput_bwd_counts.pop(mb, None)

    def clear_minibatch_state(self):
        """
        Clear minibatch state before start of a new minibatch
        """
        self._to_recv_real.clear()
        self._to_recv_dummy.clear()
        self._to_recv_sequential.clear()

        self._to_send_dummy.clear()
        self._to_send_count.clear()

        self._module_outputs.clear()
        self._pending_bwd_counts.clear()
        self._smpinput_bwd_counts.clear()

    def clear_tensor_parallelism_modules(self):
        # Remove references for the origin modules that are replaced by tp counterparts
        for m in self._tensor_parallelism_modules:
            for child in m.modules():
                if child in self._module_partitions:
                    del self._module_partitions[child]
            if m in self._module_partitions:
                del self._module_partitions[m]
        self._tensor_parallelism_modules.clear()
        self._tensor_parallelism_config.clear()

    def assign_partition(self, module: nn.Module, partition: Optional[int] = None):
        # this is called inside a init call, for some reason cant print module object here
        if partition is None:
            partition = self._cur_partition
        self._module_partitions[module] = partition

    def get_partition(self, module: nn.Module):
        return self._module_partitions[module]

    def should_tensor_parallelize(self, module):
        return module in self._tensor_parallelism_modules

    def register_distributed(self, module: nn.Module):
        """Mark the module and all its descendants as distributed/tensor-parallelized."""

        self._distributed_modules.add(module)
        for c in module.children():
            self.register_distributed(c)

    def is_distributed(self, module: nn.Module):
        return module in self._distributed_modules

    def should_checkpoint_activations(self, module):
        return module in self._activation_checkpoint_modules_config

    def get_checkpoint_activations_config(self, module):
        return self._activation_checkpoint_modules_config[module]

    def get_tp_config(self, module):
        return self._tensor_parallelism_config.get(module, {})

    def check_module_partition(self, module):
        rank = None
        for m in module.modules():
            if rank == None:
                rank = self.get_partition(m)
            else:
                if rank != self.get_partition(m):
                    return False
        return True

    def maybe_mark_for_tensor_parallelism(self, module: nn.Module):
        """If tensor parallelism is currently enabled and the module is supported,
        mark the module for tensor parallelism."""

        from smdistributed.modelparallel.torch.state_mod import state

        # not using isinstance because sub-classes of supported modules may not be supported
        if self._tensor_parallelism_enabled and state.tp_registry.is_supported(type(module)):
            self._tensor_parallelism_modules.add(module)
            self._tensor_parallelism_config[module] = self._current_tp_config

    def set_activation_checkpointing(
        self, module, preserve_rng_state=True, pack_args_as_tuple=False, strategy="each"
    ):
        from smdistributed.modelparallel.torch.state_mod import state

        if pack_args_as_tuple:
            logger.warning(
                "pack_args_as_tuple argument is deprecated, and will be removed in a future version of smp. This argument is a no-op and is not required."
            )

        if not state.model:
            raise DistributedModelNotWrappedError(
                "set_activation_checkpointing can be called on a module"
            )

        """ Enable activation checkpointing for the given module. """
        if not isinstance(module, nn.Module):
            raise CheckpointingConfigError(
                "Only a module of type nn.Module can be passed for activation checkpointing"
            )
        if not isinstance(module, nn.Sequential):
            if strategy != "each":
                # each is just the default, if user tries to change this, we throw error
                raise CheckpointingConfigError(
                    "strategy can only be used when checkpointing Sequential modules"
                )

        if state.cfg.zero2d_enabled():
            module_params = {p for p in module.parameters()}

            # current limitation with zero2d: modules that share parameters cannot be checkpointed - causes accuracy issues. TODO: fix
            if any((len(state.model.get_module_for_param(p)) > 1) for p in module_params):
                raise CheckpointingConfigError(
                    f"When sharded data parallelism is enabled, modules that share parameters cannot be checkpointed. Offending module: {self._module_to_name[module]}."
                )
        self._activation_checkpoint_modules_config[module] = CheckpointConfig(
            enabled=True,
            preserve_rng_state=preserve_rng_state,
            module_name=self.get_module_name(module),
            strategy=strategy,
        )

    def set_tensor_parallelism(self, module, enabled=True, **tp_config):
        """Enable or disable tensor parallelism for the given module. If disabling, disable for the entire sub-tree.
        If enabling, it is enabled for the top-most level supported modules only."""
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model:
            raise DistributedModelWrappedError("using the set_tensor_parallelism API")

        if not enabled:
            # if disabling, disable for the entire subtree
            if module in self._tensor_parallelism_modules:
                name = module.__class__.__name__
                logger.warning(
                    f"Disabling previously-enabled tensor parallelism for module of type {name}."
                )
            self._tensor_parallelism_modules.discard(module)
            for c in module.children():
                self.set_tensor_parallelism(c, False)
        else:
            # if enabling, enable for the topmost-level supported modules only
            stack = [module]
            visited = set()
            while len(stack) > 0:
                m = stack.pop()
                if m not in visited:
                    visited.add(m)
                    if state.tp_registry.is_supported(type(m)):
                        self._tensor_parallelism_modules.add(m)
                        self._tensor_parallelism_config[m] = tp_config
                    else:
                        stack.extend([c for c in m.children()])

    def _verify_partition_info(self, partition_info):
        loaded_modules = set(partition_info.keys())
        existing_modules = set(self._name_to_module.keys())
        extra_loaded = loaded_modules.difference(existing_modules)
        missing_existing = existing_modules.difference(loaded_modules)
        missing_existing_with_params = [
            x for x in missing_existing if len(list(self._name_to_module[x].parameters())) > 0
        ]
        if len(missing_existing_with_params) > 0:
            raise SMPCheckpointError(
                f"Error: Loading a checkpoint with extra modules names {extra_loaded} and missing module names {missing_existing}. Please check if you are loading checkpoint for the same model"
            )
        if len(missing_existing) > 0:
            logger.info(
                "Loaded checkpoint does not contain some modules which are in the current model {missing_existing}, but ignoring this since those modules have no parameters."
            )
        if len(extra_loaded) > 0:
            logger.warning(
                "Loaded checkpoint contains extra keys that are not in the model {extra_loaded}, please verify you are loading the checkpoint for the same model."
            )

    @contextmanager
    def tensor_parallelism(self, enabled=True, **tp_config):
        """
        Context manager for manual tensor parallellism. If enabled=True, tensor parallelism will
        be applied to any supported Module object created within this context, unless there is
        and inner context manager that sets enabled=False.
        """
        from smdistributed.modelparallel.torch.state_mod import state

        if not state.initialized:
            yield
        else:
            _prev_state = self._tensor_parallelism_enabled
            _prev_config = self._current_tp_config
            self._tensor_parallelism_enabled = enabled
            self._current_tp_config.update(tp_config)
            try:
                yield
            finally:
                self._tensor_parallelism_enabled = _prev_state
                self._current_tp_config = _prev_config

    def simplify_tensor_parallelism_modules(self, model):
        """If a module is marked for tensor parallelism, unmark all its descendants for tensor parallelism. Also unmark
        modules that share parameters with other modules."""

        params_to_modules = defaultdict(lambda: set())

        # schema: (module, ancestor_marked_for_tp), where the latter is an ancestor module that is marked for tp

        # traverse the modules dfs
        stack = [(model, None)]
        visited = set()
        while len(stack) > 0:
            module, ancestor_marked_for_tp = stack.pop()
            if module not in visited:
                visited.add(module)

                if ancestor_marked_for_tp is not None:
                    topmost_module_marked_for_tp = ancestor_marked_for_tp
                    if module in self._tensor_parallelism_modules:
                        # remove since an ancestor is already marked
                        self._tensor_parallelism_modules.remove(module)
                elif module in self._tensor_parallelism_modules:
                    topmost_module_marked_for_tp = module
                else:
                    topmost_module_marked_for_tp = None

                for p in module.parameters(recurse=False):
                    params_to_modules[p].add((module, topmost_module_marked_for_tp))

                stack.extend([(c, topmost_module_marked_for_tp) for c in module.children()])

        # unmark for tensor parallelism if sharing parameters with another module
        for param, modules_marks in params_to_modules.items():
            tp_ancestors = set([a for _, a in modules_marks])
            if len(tp_ancestors) > 1:
                # there are multiple, distinct, distributed modules sharing parameters
                # disabling tp for all
                for m, ancestor_marked_for_tp in modules_marks:
                    if ancestor_marked_for_tp is not None:
                        logger.warning(
                            f"Disabling tensor parallelism for module of type {type(ancestor_marked_for_tp)} since it shares parameters with another module."
                        )
                        self._tensor_parallelism_modules.discard(ancestor_marked_for_tp)

    @contextmanager
    def partition(self, i: int):
        """
        Context manager to help with manual assignment. A Module object created within this context is assigned the partition i
        """
        from smdistributed.modelparallel.torch.state_mod import state

        if not state.initialized or self.cfg.auto_partition:
            yield
        else:
            if i < 0 or i >= self.cfg.pipeline_parallel_degree:
                raise InvalidPartitionIDError

            _prev_partition = self._cur_partition
            self._cur_partition = i
            try:
                yield
            finally:
                self._cur_partition = _prev_partition

    def save_input_size(self, module: nn.Module, size: int):
        self._traced_input_sizes[module] = size

    def save_output_size(self, module: nn.Module, size: int):
        self._traced_output_sizes[module] = size

    def record_execution_order(self, module: nn.Module):
        self._module_execution_order.append(module)

    def name_modules_and_create_parent_map(self):
        """
        Converts the key in tracing information dictionaries from module object to a string identifier.
        The format of this name is a '/' separated list of module names. The main model is named 'main'.
        Any child module is named with the variable name used for that module in code.
        For example,
        class A1:
            self.b = A2()

        class A2:
            self.c = A3()

        model = A1()

        A1 -> main
        A2 -> main/b
        A3 -> main/b/c
        """
        # local import as module manager itself is a member of state and causes circular imports
        from smdistributed.modelparallel.torch.state_mod import state

        if not self._module_to_name:
            main_module = state.model
            self._module_to_name[main_module] = "main"
            self._name_to_module["main"] = main_module

            def record_child_module_name(parent: nn.Module):
                # sort for deterministic ordering
                named_children = sorted(list(parent.named_children()), key=lambda x: x[0])
                for n, c in named_children:
                    child_name = os.path.join(self._module_to_name[parent], n)
                    if c not in self._module_to_name:
                        self._module_to_name[c] = child_name
                        self._name_to_module[child_name] = c
                        # if a module is held by more than one parent module, there can be more than
                        # one path from root to that module, store first name for now
                        # since this is only for serialization and deserialization,
                        # we only need consistent name across all
                    self._parent_map[self._module_to_name[c]].append(self._module_to_name[parent])
                    record_child_module_name(c)

            self._parent_map[self._module_to_name[main_module]] = None
            record_child_module_name(main_module)

    def get_module_name(self, module: nn.Module):
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        try:
            return self._module_to_name[module]
        except KeyError:
            raise MissingModuleError(module)

    def get_module(self, name: str):
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        return self._name_to_module[name]

    def get_module_names(self):
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        return self._module_to_name.values()

    def get_serialized_partitioning_and_tracing_states(self) -> PartitioningAndTraceResults:
        """
        Refer comment in name_modules.
        Used by the sender of rank which did the tracing while sending to other ranks.
        """
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()

        mod_partitions_by_names = {}
        mod_exc_order_by_names = []
        traced_output_sizes_by_names = {}
        traced_input_sizes_by_names = {}
        mod_exc_times_by_names = {}
        mod_mem_usage_by_names = {}
        for m, n in self._module_to_name.items():
            mod_partitions_by_names[n] = self.get_partition(m)
        for m in self._module_execution_order:
            mod_exc_order_by_names.append(self.get_module_name(m))
        for k, s in self._traced_output_sizes.items():
            traced_output_sizes_by_names[self.get_module_name(k)] = s
        for k, s in self._traced_input_sizes.items():
            traced_input_sizes_by_names[self.get_module_name(k)] = s
        for k, s in self._module_execution_times.items():
            mod_exc_times_by_names[self.get_module_name(k)] = s
        for k, s in self._module_memory_usage.items():
            mod_mem_usage_by_names[self.get_module_name(k)] = s

        data = PartitioningAndTraceResults(
            mod_partitions_by_names,
            mod_exc_order_by_names,
            traced_input_sizes_by_names,
            traced_output_sizes_by_names,
            mod_exc_times_by_names,
            mod_mem_usage_by_names,
        )
        return data

    def load_partitioning_and_trace_results(self, trace_results: PartitioningAndTraceResults):
        """
        Ranks other than the one which did the tracing receive tracing information and load them.
        """
        if not self._module_to_name or not self._name_to_module:
            self.name_modules_and_create_parent_map()

        # TODO: raise user friendly message when wrong results are loaded
        (
            mod_partitions,
            mod_exc_order,
            traced_input_sizes,
            traced_output_sizes,
            mod_exc_times,
            mod_mem_usage,
        ) = trace_results
        for m in mod_partitions:
            try:
                self._module_partitions[self.get_module(m)] = mod_partitions[m]
            except KeyError:
                # skip modules not part of original model, as this rank hasn't seen those modules
                # these will be treated as partition: None anyway
                pass

        for m in mod_exc_order:
            try:
                self._module_execution_order.append(self.get_module(m))
            except KeyError:
                pass

        for n, s in traced_input_sizes.items():
            try:
                self._traced_input_sizes[self.get_module(n)] = s
            except KeyError:
                pass

        for n, s in traced_output_sizes.items():
            try:
                self._traced_output_sizes[self.get_module(n)] = s
            except KeyError:
                pass

        for n, s in mod_exc_times.items():
            try:
                self._module_execution_times[self.get_module(n)] = s
            except KeyError:
                pass

        for n, s in mod_mem_usage.items():
            try:
                self._module_memory_usage[self.get_module(n)] = s
            except KeyError:
                pass
