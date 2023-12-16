"""Train GPT or BLOOM models."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../transformers/"))

# pylint: disable=import-error,wrong-import-order,wrong-import-position
import arguments
import translate_args
from logging_utils import get_logger

# pylint: enable=import-error

_logger = get_logger()
_CONFIG_FILE = "config_file"


class DummyObject:  # pylint: disable=too-few-public-methods
    """Dummy object."""


def get_parser():
    """Get parser for args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="", nargs="?", const="", help="Which config file to use."
    )

    return parser


def parse_args():
    """Get args from either the command line or from a config file (`--config_file`)."""
    train_args, unknown_args = arguments.parse_args()

    # Case 1: No config file is provided, fall back to `train.parse_args`.
    config_file = get_parser().parse_known_args(unknown_args)[0].config_file
    if not config_file:
        _logger.info("Using `arguments.parse_args` as config_file = `%s`.", config_file)
        return train_args

    # Case 2: Config file will be translated to `train.parse_args` and validated for all fields.
    _logger.info("Using `main.parse_args`: %s.", config_file)

    config_args = translate_args.translate_job_args(config_file)
    for name in (
        "clean_cache",  # Global
        "enable_memory_profiling",  # Global
        ### New fields.
        "checkpoint_type",
        "data_num_workers",
        "dataset_type",
        "delayed_param",
        "distributed_backend",
        "framework",
        "grad_clip",
        "hf_pretrained_model_name_or_dir",
        "logging_freq_for_avg",
        "patch_neox_rope",
        "resume_from_checkpoint",
        "save_final_model",
        "use_smp_flash_attn",
        "use_smp_implementation",
    ):
        config_args.update(
            {
                name: getattr(train_args, name),
            }
        )

    # Validates all `train_args` fields are present in `config_args`.
    train_arg_set = set(train_args.__dict__.keys())
    missing_keys = train_arg_set.difference(config_args.keys())
    if missing_keys:
        raise ValueError(
            f"Please make sure all fields are translated from config file: {sorted(missing_keys)}."
        )

    obj = DummyObject()
    for key, value in sorted(config_args.items()):
        # Skip unused ones.
        if key in train_arg_set:
            setattr(obj, key, value)

    return obj
