import os
import json


def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    return presets + [f[:f.index(".json")] for f in os.listdir(preset_folder) if f.endswith('.json')]


def get_model_filenames(folder_paths, extensions=None, name_filter=None):
    if extensions is None:
        extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']
    files = []

    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder in folder_paths:
        files += get_files_from_folder(folder, extensions, name_filter)

    return files


def makedirs_with_log(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f'Directory {path} could not be created, reason: {error}')


def get_files_from_folder(folder_path, extensions=None, name_filter=None):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, _, files in os.walk(folder_path, topdown=False):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in sorted(files, key=lambda s: s.casefold()):
            _, file_extension = os.path.splitext(filename)
            if (extensions is None or file_extension.lower() in extensions) and (name_filter is None or name_filter in _):
                path = os.path.join(relative_path, filename)
                filenames.append(path)

    return filenames