#!/usr/bin/env python3
import argparse
import yaml

def get_nested_value(config, key_path):
    """Traverse the config dict using a list of keys from key_path."""
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except KeyError:
        raise KeyError(f"Key path '{key_path}' not found in the configuration.")

def main():
    parser = argparse.ArgumentParser(
        description="Parse a YAML configuration file and extract a nested key value."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--key",
        required=True,
        help="Nested key to extract (use dot notation, e.g., metadata.output_path)."
    )
    
    args = parser.parse_args()

    try:
        with open(args.file, 'r') as stream:
            config = yaml.safe_load(stream)
    except Exception as err:
        print(f"Error reading YAML file: {err}")
        return 1

    try:
        value = get_nested_value(config, args.key)
        print(value)
    except KeyError as err:
        print(err)
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
