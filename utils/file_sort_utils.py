import os
import json

class StyleSorter:
    all_styles = []


    def try_load_sorted_styles(self, style_names, default_selected):
        self.all_styles = style_names

        try:
            if os.path.exists('sorted_styles.json'):
                with open('sorted_styles.json', 'rt', encoding='utf-8') as fp:
                    sorted_styles = []
                    for x in json.load(fp):
                        if x in all_styles:
                            sorted_styles.append(x)
                    for x in all_styles:
                        if x not in sorted_styles:
                            sorted_styles.append(x)
                    all_styles = sorted_styles
        except Exception as e:
            print('Load style sorting failed.')
            print(e)

        unselected = [y for y in all_styles if y not in default_selected]
        self.all_styles = default_selected + unselected

        return


    def sort_styles(self, selected):
        unselected = [y for y in self.all_styles if y not in selected]
        sorted_styles = selected + unselected
        try:
            with open('sorted_styles.json', 'wt', encoding='utf-8') as fp:
                json.dump(sorted_styles, fp, indent=4)
        except Exception as e:
            print('Write style sorting failed.')
            print(e)
        self.all_styles = sorted_styles
        return sorted_styles




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