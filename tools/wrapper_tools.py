import argparse
import yaml
import sys
import re
from tqdm import tqdm
import requests
import hashlib

def extract_defaults_from_trainSpeakerNet(file_path):
    """
    Extracts argument names and default values from a Python file that uses argparse.
    If no default value is specified, it defaults to boolean False.

    Args:
    file_path (str): The path to the file.

    Returns:
    argparse.Namespace: An args object with each argument set to its default value or False.
    """
    # Reading the file content
    with open(file_path, 'r', encoding='ascii') as file:
        file_content = file.read()

    # Regex patterns
    regex_pattern_full_line = r'^parser\.add_argument\(.*$'
    regex_pattern_arg_name = r"^parser\.add_argument\('--([^']*)|\"--([^\"]*)"
    regex_pattern_type = r"type=(\w+)"
    regex_pattern_default = r"default=([^,\)]*)"

    # Extracting full lines that define arguments
    full_lines = re.findall(regex_pattern_full_line, file_content, re.MULTILINE)

    # Creating the args object
    args = argparse.Namespace()

    for line in full_lines:
        # Extracting argument name
        name_match = re.search(regex_pattern_arg_name, line)
        arg_name = name_match.group(1) if name_match.group(1) else name_match.group(2)

        # Extracting type
        type_match = re.search(regex_pattern_type, line)
        arg_type = type_match.group(1) if type_match else "bool"  # Defaulting to boolean type if not specified

        # Extracting default value
        default_match = re.search(regex_pattern_default, line)
        default_value = default_match.group(1) if default_match else 'False'

        # Casting the default value to the specified type
        if arg_type == 'int':
            casted_default = int(default_value) if default_value.isdigit() else 0
        elif arg_type == 'float':
            casted_default = float(default_value) if default_value.replace('.', '', 1).isdigit() else 0.0
        elif arg_type == 'bool':
            casted_default = default_value.lower() in ('true', '1', 'yes') if default_value not in [None, 'False'] else False
        elif arg_type == 'str':
            casted_default = default_value.strip('"').strip("'") if default_value not in [None, 'None'] else None
        else:
            casted_default = default_value  # For unrecognized types, keep as-is

        # Setting the attribute
        setattr(args, arg_name, casted_default)

    return args

def update_args_from_config(args, config_file_path):
    """
    Updates the args object with values from the YAML config file.

    Args:
    args (argparse.Namespace): The initial args object.
    config_file_path (str): Path to the YAML config file.

    Returns:
    argparse.Namespace: Updated args object.
    """
    # Reading the config file
    with open(config_file_path, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)

    # Updating args based on config
    for k, v in yml_config.items():
        if hasattr(args, k):
            current_value = getattr(args, k)
            typ = type(current_value)

            # Casting the value to the correct type
            if typ == int:
                casted_value = int(v)
            elif typ == float:
                casted_value = float(v)
            elif typ == bool:
                casted_value = v.lower() in ('true', '1', 'yes') if isinstance(v, str) else bool(v)
            elif typ == str:
                casted_value = str(v)
            else:
                casted_value = v  # Keep the value as-is for unrecognized types

            setattr(args, k, casted_value)
        else:
            sys.stderr.write(f"Ignored unknown parameter {k} in yaml.\n")

    args.config = config_file_path

    return args

def generate_args_with_config(trainSpeakerNet_path, config_path=None):
    """
    Processes the arguments from the trainSpeakerNet.py file and optionally updates them
    with values from a provided YAML config file.

    Args:
    trainSpeakerNet_path (str): The path to the trainSpeakerNet.py file.
    config_path (str, optional): The path to the YAML config file. If None, no update is performed.

    Returns:
    argparse.Namespace: The updated args object.
    """
    # Extracting defaults and types from the trainSpeakerNet file
    args = extract_defaults_from_trainSpeakerNet(trainSpeakerNet_path)

    # If a config path is provided, update the args
    if config_path is not None:
        args = update_args_from_config(args, config_path)

    return args


def download_file_and_check_sha256(url, expected_hash=None):
    """
    Downloads a file from the given URL using requests and checks its SHA256 hash,
    displaying a simple progress bar.

    Args:
    url (str): The URL of the file to be downloaded.
    expected_hash (str, optional): The expected SHA256 hash of the file.

    Returns:
    str: The SHA256 hash of the downloaded file.
    bool: True if the hash matches the expected hash, False otherwise.
    """
    # Extracting the file name from the URL
    file_name = url.split('/')[-1]

    # Sending GET request and streaming the content
    response = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    # Check if file is downloaded completely
    if total_size != 0 and progress_bar.n != total_size:
        raise Exception("Error, downloaded file is incomplete.")

    # Compute the SHA256 hash of the file
    sha256_hash = hashlib.sha256()
    with open(file_name, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()

    # Compare with the expected hash
    hash_match = (file_hash == expected_hash) if expected_hash else True

    return file_hash, hash_match