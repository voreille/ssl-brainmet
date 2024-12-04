from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def build_model_from_plans(
    plans_path: Path,
    dataset_json_path: Path,
    configuration: str = "3d_fullres",
    deep_supervision: bool = True,
):
    """
    Builds and returns a model based on the nnUNet plans file.

    Args:
        plans_path (Path): Path to the nnUNetPlans.json file.
        configuration (str): Configuration name to use from the plans. Default is "default".

    Returns:
        torch.nn.Module: The model built based on the plans.
    """
    plans_path = Path(plans_path)  # Ensure compatibility with Path
    if not plans_path.exists():
        raise ValueError(f"Invalid plans path: {plans_path}")

    # Load the plans file
    plans = load_json(str(plans_path))
    dataset_json = load_json(dataset_json_path)
    plans_manager = PlansManager(plans)

    # Get the configuration manager
    config_manager = plans_manager.get_configuration(configuration)

    # Determine the number of input channels
    num_input_channels = determine_num_input_channels(
        plans_manager,
        config_manager,
        dataset_json,
    )
    label_manager = plans_manager.get_label_manager(dataset_json)
    # Build and return the network
    return get_network_from_plans(
        config_manager.network_arch_class_name,
        config_manager.network_arch_init_kwargs,
        config_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        allow_init=True,
        deep_supervision=deep_supervision,
    )


# Example Usage
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]

    model = build_model_from_plans(
        project_dir / "ssl_brainmet/config/nnUNetPlans.json",
        project_dir / "ssl_brainmet/config/dataset.json",
        configuration="3d_fullres",
        deep_supervision=True,
    )
    print(model)
