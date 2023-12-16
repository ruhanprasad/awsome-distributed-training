"""Unit test for GPTNeoX config files."""

# pylint: disable=wrong-import-position
# Standard Library
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../transformers/"))

import unittest
from typing import Tuple

# Third Party
from google.protobuf import text_format

# pylint: disable=import-error,no-name-in-module
from parameterized import parameterized

# First Party
from proto import job_pb2, jobs_pb2

# pylint: enable=import-error,no-name-in-module


#
# GPT NeoX config files.
#

# Consistent with ./7b.sh
_CONFIG_FILE = "gpt_neox/fsdp_gpt_neox_007b.pbtxt"
_CONFIG_FILE_007B_COPY_SELF = "./configs/fsdp_gpt_neox_007b.pbtxt"
_CONFIG_FILE_007B_COPY = "./configs/fsdp_gpt_neox_007b_matching_065b.pbtxt"
_CONFIG_FILE_DEMO = "gpt_neox/fsdp_gpt_neox_007b_demo_new.pbtxt"

# 013b.pbtxt is copied from ../../transformers/script/
_CONFIG_FILE_DEMO_V1_007B = "gpt_neox/fsdp_gpt_neox_007b_demo.pbtxt"
_CONFIG_FILE_DEMO_V1_013B = "gpt_neox/fsdp_gpt_neox_013b_demo.pbtxt"

# TODO(sliuxl): Add those files.
_CONFIG_FILE_CONVERGENCE_DEMO = "gpt_neox/fsdp_gpt_neox_007b_convergence_demo.pbtxt"
_CONFIG_FILE_CONVERGENCE_RERUN_DEMO = "gpt_neox/fsdp_gpt_neox_007b_convergence_rerun_demo.pbtxt"

# Consistent with ./65b.sh
_CONFIG_FILE_065B = "gpt_neox/fsdp_gpt_neox_065b.pbtxt"
_CONFIG_FILE_065B_COPY = "./configs/fsdp_gpt_neox_065b.pbtxt"


#
# Llama config files.
#

# Consistent with ./llama_v2_7b.sh
_LLAMA_CONFIG_FILE_007B = "llama_v2/fsdp_llama_v2_007b.pbtxt"

_LLAMA_CONFIG_FILE_070B = "llama_v2/fsdp_llama_v2_070b.pbtxt"
_LLAMA_CONFIG_FILE_070B_COPY = "./configs/fsdp_llama_v2_070b.pbtxt"


# pylint: disable=no-member
def _parse_proto(filename: str, proto_type=job_pb2.Job):
    """Parse config from a proto file."""
    with open(filename, "r") as ifile:  # pylint: disable=unspecified-encoding
        return text_format.Parse(ifile.read(), proto_type())


def _clear(proto: job_pb2.Job, field: str):
    """Clear (nested) field in proto: in place."""
    fields = field.split(".")
    for index, name in enumerate(fields):
        if index == len(fields) - 1:
            proto.ClearField(name)
        else:
            proto = getattr(proto, name)


class TestConfigFiles(unittest.TestCase):
    """Unit test for config files."""

    @parameterized.expand(
        (
            "configs/model__gpt_neox__007b_native.pbtxt",
            "configs/model__gpt_neox__007b.pbtxt",
            "configs/model__gpt_neox__007b-sdp16.pbtxt",
            "configs/model__gpt_neox__065b_native.pbtxt",
            "configs/model__gpt_neox__065b.pbtxt",
            "configs/model__gpt_neox__065b-sdp256.pbtxt",
            "configs/model__llama_v2__007b_native.pbtxt",
            "configs/model__llama_v2__007b.pbtxt",
            "configs/model__llama_v2__007b-sdp016.pbtxt",
            "configs/model__llama_v2__007b-sdp032.pbtxt",
            "configs/model__llama_v2__007b-sdp064.pbtxt",
            "configs/model__llama_v2__007b-sdp128.pbtxt",
            "configs/model__llama_v2__007b-sdp256.pbtxt",
            "configs/model__llama_v2__013b_native.pbtxt",
            "configs/model__llama_v2__013b.pbtxt",
            "configs/model__llama_v2__013b-sdp016.pbtxt",
            "configs/model__llama_v2__013b-sdp032.pbtxt",
            "configs/model__llama_v2__013b-sdp064.pbtxt",
            "configs/model__llama_v2__013b-sdp128.pbtxt",
            "configs/model__llama_v2__013b-sdp256.pbtxt",
            "configs/model__llama_v2__070b_native.pbtxt",
            "configs/model__llama_v2__070b.pbtxt",
            "configs/model__llama_v2__070b-sdp256.pbtxt",
            "configs/model__llama_v2__070b-sdp512.pbtxt",
            "demos/demo.pbtxt",
            "demos/demo_activation_checkpointing.pbtxt",
            "demos/demo_activation_offloading.pbtxt",
        )
    )
    def test_config_file(self, config_file: str):
        """Unit test for valid config files."""
        config = _parse_proto(config_file)
        self.assertIsInstance(config, job_pb2.Job)

    def _verify_config_file_comparison(
        self,
        lhs: job_pb2.Job,
        rhs: job_pb2.Job,
        fields: Tuple[str],
    ):
        """Unit test to compare config files."""
        for field in fields:
            _clear(lhs, field)
            _clear(rhs, field)

        self.assertEqual(lhs, rhs)

    @parameterized.expand(
        (
            #
            # GPT-NeoX
            #
            # Compare with `_CONFIG_FILE`.
            (
                _CONFIG_FILE,
                _CONFIG_FILE_007B_COPY_SELF,
                (
                    "rubik_job.model.checkpoints.checkpoints_dir",
                    "rubik_job.model.max_total_steps",
                    "rubik_job.fsdp.apply_activation_checkpoint",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                _CONFIG_FILE,
                _CONFIG_FILE_DEMO,
                (
                    "name",
                    "rubik_job.model.checkpoints.checkpoints_dir",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.max_total_steps",
                ),
            ),
            (
                _CONFIG_FILE_DEMO,
                "gpt_neox/fsdp_gpt_neox_007b_demo_nsys.pbtxt",
                (
                    "name",
                    "rubik_job.model.nsys_profiling",
                    "runtime.mpi_cmd.system_cmd_args",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                _CONFIG_FILE_DEMO,
                _CONFIG_FILE_DEMO_V1_007B,
                (
                    "name",
                    "rubik_job.fsdp.apply_activation_checkpoint",
                    "rubik_job.fsdp.forward_prefetch",
                    "rubik_job.fsdp.limit_all_gathers",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "rubik_job.fsdp.sharding_strategy",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.shared_data.batch_size",
                    "rubik_job.model.transformer_model.gpt_neox.vocab_size",
                    "rubik_job.model.val_data.batch_size",
                ),
            ),
            (
                _CONFIG_FILE_065B,
                _CONFIG_FILE_065B_COPY,
                (
                    "rubik_job.model.checkpoints.checkpoints_dir",
                    "rubik_job.model.max_total_steps",
                ),
            ),
            # Different model size.
            (
                _CONFIG_FILE,
                _CONFIG_FILE_065B,
                (
                    "name",
                    "rubik_job.fsdp.apply_activation_checkpoint",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "rubik_job.fsdp.use_orig_params",
                    "rubik_job.model.checkpoints.checkpoint_freq",
                    "rubik_job.model.transformer_model.gpt_neox.hidden_size",
                    "rubik_job.model.transformer_model.gpt_neox.initializer_range",
                    "rubik_job.model.transformer_model.gpt_neox.num_attention_heads",
                    "rubik_job.model.transformer_model.gpt_neox.num_hidden_layers",
                    "rubik_job.model.validation_freq",
                    "runtime.mpi_cmd.system_cmd_pre",
                    "runtime.nodes.num_nodes",
                ),
            ),
            (
                _CONFIG_FILE_007B_COPY,
                _CONFIG_FILE_065B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "rubik_job.model.transformer_model.gpt_neox.hidden_size",
                    "rubik_job.model.transformer_model.gpt_neox.initializer_range",
                    "rubik_job.model.transformer_model.gpt_neox.num_attention_heads",
                    "rubik_job.model.transformer_model.gpt_neox.num_hidden_layers",
                    "runtime.nodes.num_nodes",
                ),
            ),
            (
                _CONFIG_FILE_DEMO_V1_007B,
                _CONFIG_FILE_DEMO_V1_013B,
                (
                    "name",
                    "rubik_job.model.transformer_model.gpt_neox.hidden_size",
                    "rubik_job.model.transformer_model.gpt_neox.num_attention_heads",
                    "rubik_job.model.transformer_model.gpt_neox.num_hidden_layers",
                ),
            ),
            #
            # Llama
            #
            (
                _LLAMA_CONFIG_FILE_007B,
                _LLAMA_CONFIG_FILE_070B,
                (
                    "name",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "rubik_job.lr_scheduler.aws_annealing_lr.start_lr",
                    "rubik_job.model.transformer_model.llama.hidden_size",
                    "rubik_job.model.transformer_model.llama.initializer_range",
                    "rubik_job.model.transformer_model.llama.intermediate_size",
                    "rubik_job.model.transformer_model.llama.num_attention_heads",
                    "rubik_job.model.transformer_model.llama.num_hidden_layers",
                    "rubik_job.model.transformer_model.llama.num_key_value_heads",
                    "rubik_job.model.transformer_model.llama.rms_norm_eps",
                    "runtime.mpi_cmd.system_cmd_pre",
                    "runtime.nodes.num_nodes",
                ),
            ),
            (
                _LLAMA_CONFIG_FILE_070B,
                _LLAMA_CONFIG_FILE_070B_COPY,
                ("rubik_job.model.checkpoints.checkpoints_dir",),
            ),
        )
    )
    def test_config_file_comparison(
        self,
        config_file_lhs: str,
        config_file_rhs: str,
        fields: Tuple[str],
    ):
        """Unit test to compare config files."""
        lhs = _parse_proto(config_file_lhs)
        rhs = _parse_proto(config_file_rhs)

        self._verify_config_file_comparison(lhs, rhs, fields)

    def _test_config_file_convergence_demo(self):
        """Sanity check for the convergence job config file."""
        config = _parse_proto(_CONFIG_FILE_CONVERGENCE_DEMO)

        # Needs to load from disk for reruns if possible.
        self.assertTrue(config.rubik_job.model.checkpoints.load_partial)

        # Needs to have extra checkpoints:
        #   - Non empty
        #   - To a different dir
        #   - With larger interval
        #   - Keeping all checkpoints
        self.assertGreater(len(config.rubik_job.model.checkpoints_extra), 0)
        self.assertNotEqual(
            config.rubik_job.model.checkpoints_extra[0].checkpoints_dir,
            config.rubik_job.model.checkpoints.checkpoints_dir,
        )
        self.assertGreater(
            config.rubik_job.model.checkpoints_extra[0].checkpoint_freq,
            config.rubik_job.model.checkpoints.checkpoint_freq,
        )
        self.assertLessEqual(config.rubik_job.model.checkpoints_extra[0].max_num_checkpoints, 0)

    def _test_config_file_convergence_rerun_demo(self):
        """Sanity check for the convergence job rerun config file."""
        config = _parse_proto(_CONFIG_FILE_CONVERGENCE_RERUN_DEMO)

        # Needs to have `max_inc_steps`, instead of `max_total_steps`.
        self.assertTrue(config.rubik_job.model.HasField("max_inc_steps"))
        self.assertFalse(config.rubik_job.model.HasField("max_total_steps"))

        # Needs to have `convergence_job`.
        self.assertTrue(config.rubik_job.HasField("convergence_job"))
        if config.rubik_job.convergence_job.HasField("ckpt_interval"):
            # `max_inc_steps` should be larger than checkpoint interval for full coverage.
            self.assertLess(
                config.rubik_job.convergence_job.ckpt_interval.interval,
                config.rubik_job.model.max_inc_steps,
            )

        # Needs to have certain fields for checkpoints:
        #   - `checkpoints_dir_readonly`: The checkpoint dir
        #   - `checkpoint_partial_tag`: The tag (sub dir regex) for a given step
        self.assertTrue(config.rubik_job.model.checkpoints.HasField("checkpoints_dir_readonly"))
        self.assertTrue(config.rubik_job.model.checkpoints.HasField("checkpoint_partial_tag"))

    @parameterized.expand(
        (
            "configs/fsdp_gpt_neox_007b_benchmark.pbtxt",
            "configs/fsdp_gpt_neox_007b_benchmark_00.pbtxt",
        )
    )
    def test_valid_gpt_neox_benchmark_config_file(self, config_file: str):
        """Sanity check for GPT NeoX benchmark config file."""
        fields = (
            "name",
            "rubik_job.fsdp.apply_activation_checkpoint",
            "rubik_job.fsdp.cpu_offload",
        )

        lhs = _parse_proto(_CONFIG_FILE_007B_COPY_SELF)
        rhs = _parse_proto(config_file, proto_type=jobs_pb2.Jobs).global_base_job
        self._verify_config_file_comparison(lhs, rhs, fields)

    @parameterized.expand(
        (
            (
                "configs/fsdp_gpt_neox_007b_benchmark_02_token4m_native.pbtxt",
                _CONFIG_FILE_007B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "rubik_job.fsdp.sharding_strategy",
                ),
            ),
            (
                "configs/fsdp_gpt_neox_065b_benchmark.pbtxt",
                _CONFIG_FILE_065B_COPY,
                (
                    "name",
                ),
            ),
            (
                "configs/fsdp_gpt_neox_065b_benchmark_00_token4m.pbtxt",
                _CONFIG_FILE_065B_COPY,
                (
                    "name",
                ),
            ),
            (
                "configs/fsdp_gpt_neox_065b_benchmark_01_token4m.pbtxt",
                _CONFIG_FILE_065B_COPY,
                (
                    "name",
                ),
            ),
            (
                "configs/fsdp_gpt_neox_065b_benchmark_02_token2m.pbtxt",
                _CONFIG_FILE_065B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.shared_data.use_synthetic_data",
                    "rubik_job.model.transformer_model.gpt_neox",
                    "rubik_job.model.transformer_model.named_model",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                "configs/fsdp_gpt_neox_065b_benchmark_02_token4m.pbtxt",
                _CONFIG_FILE_065B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "rubik_job.fsdp.sharding_strategy",
                    "rubik_job.model.experiment_name",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                "configs/fsdp_gpt_neox_065b_benchmark_02_token4m_native.pbtxt",
                _CONFIG_FILE_065B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.sharded_data_parallel_degree",
                    "rubik_job.fsdp.sharding_strategy",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_00_token4m.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_01_token4m.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_02_token2m.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.use_orig_params",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.shared_data.use_synthetic_data",
                    "rubik_job.model.transformer_model.llama",
                    "rubik_job.model.transformer_model.named_model",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_02_token2m_sdp.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.use_orig_params",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.shared_data.use_synthetic_data",
                    "rubik_job.model.transformer_model.llama",
                    "rubik_job.model.transformer_model.named_model",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_02_target4m.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.use_orig_params",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.shared_data.use_synthetic_data",
                    "rubik_job.model.transformer_model.llama",
                    "rubik_job.model.transformer_model.named_model",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_02_target4m_nsys.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.use_orig_params",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.nsys_profiling",
                    "rubik_job.model.shared_data.use_synthetic_data",
                    "rubik_job.model.transformer_model.llama",
                    "rubik_job.model.transformer_model.named_model",
                    "runtime.mpi_cmd.system_cmd_args",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_02_token4m.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.use_orig_params",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.shared_data.use_synthetic_data",
                    "rubik_job.model.transformer_model.llama",
                    "rubik_job.model.transformer_model.named_model",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
            (
                "configs/fsdp_llama_v2_070b_benchmark_02_token4m_sdp.pbtxt",
                _LLAMA_CONFIG_FILE_070B_COPY,
                (
                    "name",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.fsdp.use_orig_params",
                    "rubik_job.fsdp.cpu_offload",
                    "rubik_job.model.experiment_name",
                    "rubik_job.model.shared_data.use_synthetic_data",
                    "rubik_job.model.transformer_model.llama",
                    "rubik_job.model.transformer_model.named_model",
                    "runtime.mpi_cmd.system_cmd_pre",
                ),
            ),
        )
    )
    def test_valid_benchmark_config_file(self, config_file: str, template_config_file: str, fields):
        """Sanity check for Llama benchmark config file."""

        rhs_jobs = _parse_proto(config_file, proto_type=jobs_pb2.Jobs)

        # Sanity check for job instances: Unique.
        num_job_instances = len(rhs_jobs.job_instances)
        for lhs in range(num_job_instances):
            for rhs in range(lhs + 1, num_job_instances):
                self.assertNotEqual(rhs_jobs.job_instances[lhs], rhs_jobs.job_instances[rhs])

        lhs = _parse_proto(template_config_file)
        # Sanity check for model type: Same for lhs & rhs.
        for job_instance in rhs_jobs.job_instances:
            job = job_instance.job
            if job.HasField("rubik_job") and job.rubik_job.model.HasField("transformer_model"):
                # They should be using the same model type.
                self.assertEqual(
                    lhs.rubik_job.model.transformer_model.WhichOneof("model"),
                    job.rubik_job.model.transformer_model.WhichOneof("model"),
                )

        self._verify_config_file_comparison(lhs, rhs_jobs.global_base_job, fields)


if __name__ == "__main__":
    unittest.main()
