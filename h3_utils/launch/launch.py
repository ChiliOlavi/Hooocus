import os
import sys

ROOT_DIR = os.path.abspath(__file__).split("h3_utils")[0]
sys.path.append(ROOT_DIR)

import ssl
import platform

from modules.hash_cache import init_cache, load_cache_from_file
from h3_utils.launch.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
from modules.model_loader import load_file_from_url
from h3_utils.config import HOOOCUS_VERSION, GlobalEnv, LAUNCH_ARGS
from h3_utils.flags import LORA_FILENAMES, MODEL_FILENAMES
from h3_utils.path_configs import FolderPathsConfig
from h3_utils.logging_util import LoggingUtil

log = LoggingUtil().get_logger()
args = LAUNCH_ARGS

log.info(f"Python {sys.version}")
log.info(f"Hooocus version: {HOOOCUS_VERSION}")


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]
ssl._create_default_https_context = ssl._create_unverified_context


def prepare_environment():
    hash_cache = load_cache_from_file()
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "h3_utils/requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Hooocus version: {HOOOCUS_VERSION}")

    if GlobalEnv.REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if GlobalEnv.TRY_INSTALL_XFORMERS:
        if GlobalEnv.REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.23')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if GlobalEnv.REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    
    """ DefaultConfigImageGen.base_model_name, FolderPathsConfig.path_checkpoints = download_models(
        DefaultConfigImageGen.base_model_name,
        DefaultConfigImageGen.previous_default_models,
        FolderPathsConfig.path_checkpoints,
        FolderPathsConfig.path_embeddings,
        FolderPathsConfig.path_loras,
        FolderPathsConfig.path_vae,
        args=LAUNCH_ARGS,
    ) """

    if args.gpu_device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
        print("Set device to:", args.gpu_device_id)

    if args.hf_mirror is not None:
        os.environ['HF_MIRROR'] = str(args.hf_mirror)
        print("Set hf_mirror to:", args.hf_mirror)

    os.environ["U2NET_HOME"] = FolderPathsConfig.path_inpaint

    os.environ['GRADIO_TEMP_DIR'] = FolderPathsConfig.default_temp_path

    if args.temp_path_cleanup_on_launch:
        print(f'[Cleanup] Attempting to delete content of temp dir {FolderPathsConfig.default_temp_path}')
        result = delete_folder_content(FolderPathsConfig.default_temp_path, '[Cleanup] ')
        if result:
            print("[Cleanup] Cleanup successful")
        else:
            print(f"[Cleanup] Failed to delete content of temp dir.")

    if len(hash_cache) == 0 and (len(MODEL_FILENAMES) > 0 or len(LORA_FILENAMES) > 0):
        hash_cache = init_cache(MODEL_FILENAMES, FolderPathsConfig.path_checkpoints, LORA_FILENAMES, FolderPathsConfig.path_loras)
        if len(hash_cache) > 0:
            print(f'[Cache] Initialized with {len(hash_cache)} entries.')
        else:
            print('[Cache] Initialization failed.')

    if args.rebuild_hash_cache:
        init_cache(MODEL_FILENAMES, FolderPathsConfig.path_checkpoints, LORA_FILENAMES, FolderPathsConfig.path_loras,)
        print('[Cache] Rebuilt cache.')
    return


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads, args=args):
    from modules.util import get_file_from_folder_list

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=FolderPathsConfig.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=FolderPathsConfig.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not args.always_download_new_model:
        if not os.path.isfile(get_file_from_folder_list(default_model, FolderPathsConfig.path_checkpoints)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(get_file_from_folder_list(alternative_model_name, FolderPathsConfig.path_checkpoints)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    #for file_name, url in checkpoint_downloads.items():
    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, FolderPathsConfig.path_checkpoints))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=FolderPathsConfig.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, FolderPathsConfig.path_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=FolderPathsConfig.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads
