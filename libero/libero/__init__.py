import os
import yaml

# This is a default path for localizing all the benchmark related files
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
)
config_file = os.path.join(libero_config_path, "config.yaml")

# Global variable to cache assets path
_assets_path_cache = None


def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = os.path.dirname(os.path.abspath(__file__))
    else:
        benchmark_root_path = custom_location

    # This is a default path for localizing all the default bddl files
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")

    # This is a default path for localizing all the default bddl files
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")

    # This is a default path for localizing all the default datasets
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")

    # This is a default path for localizing all the default assets
    assets_default_path = os.path.join(benchmark_root_path, "./assets")

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
        "assets": assets_default_path,
    }


def get_libero_path(query_key):
    with open(config_file, "r") as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))

    # Give warnings in case the user needs to access the paths
    # for key in config:
    #     if not os.path.exists(config[key]):
    #         print(f"[Warning]: {key} path {config[key]} does not exist!")

    assert (
        query_key in config
    ), f"Key {query_key} not found in config file {config_file}. You need to modify it. Available keys are: {config.keys()}"
    return config[query_key]


def set_libero_default_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    print(
        f"[Warning] You are changing the default path for Libero config. This will affect all the paths in the config file."
    )
    new_config = get_default_path_dict(custom_location)
    with open(config_file, "w") as f:
        yaml.dump(new_config, f)


if not os.path.exists(libero_config_path):
    os.makedirs(libero_config_path)

def get_assets_path():
    """
    Get the path to assets, either from HuggingFace Hub or local installation.
    
    Returns:
        str: Path to the assets directory
    """
    global _assets_path_cache
    
    # Return cached path if available
    if _assets_path_cache is not None and os.path.exists(_assets_path_cache):
        return _assets_path_cache
    
    # First try to use local assets if they exist
    local_assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    if os.path.exists(local_assets_path):
        _assets_path_cache = local_assets_path
        return local_assets_path
    
    # Otherwise, download from HuggingFace Hub
    try:
        from libero.libero.utils.download_utils import download_assets_from_huggingface
        print("Local assets not found. Downloading from HuggingFace Hub...")
        assets_path = download_assets_from_huggingface()
        _assets_path_cache = assets_path
        return assets_path
    except Exception as e:
        print(f"Warning: Could not download assets from HuggingFace Hub: {e}")
        # Fallback to local path even if it doesn't exist (will cause error later if needed)
        return local_assets_path


if not os.path.exists(config_file):
    # Create a default config file

    default_path_dict = get_default_path_dict()
    answer = input(
        "Do you want to specify a custom path for the dataset folder? (Y/N): "
    ).lower()
    if answer == "y":
        # If the user wants to specify a custom storage path, prompt them to enter it
        custom_dataset_path = input(
            "Enter the path where you want to store the datasets: "
        )
        full_custom_dataset_path = os.path.join(
            os.path.abspath(os.path.expanduser(custom_dataset_path)), "datasets"
        )
        # Check if the custom storage path exists, and create if it doesn't

        print("The full path of the custom storage path you entered is:")
        print(full_custom_dataset_path)
        print("Do you want to continue? (Y/N)")
        confirm_answer = input().lower()
        if confirm_answer == "y":
            if not os.path.exists(full_custom_dataset_path):
                os.makedirs(full_custom_dataset_path)
            default_path_dict["datasets"] = full_custom_dataset_path
    print("Initializing the default config file...")
    print(f"The following information is stored in the config file: {config_file}")
    # write all the paths into a yaml file
    with open(config_file, "w") as f:
        yaml.dump(default_path_dict, f)
    for key, value in default_path_dict.items():
        print(f"{key}: {value}")
