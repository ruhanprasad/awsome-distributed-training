"""Train GPT or BLOOM models."""

# pylint: disable=import-error
from sm_env_utils import enable_dummy_sm_env
# Set dummy SageMaker env var if not set to pass guardrail
# for Rubik and Herring cluster scripts.
enable_dummy_sm_env()  # needs to be called before torch sagemaker is imported
import main_lib
import train_lib
# pylint: enable=import-error


def main():
    """Main function to train GPT."""
    args = main_lib.parse_args()
    train_lib.main(args)


if __name__ == "__main__":
    main()
