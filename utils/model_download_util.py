
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.model_loader import load_file_from_url
from utils.consts import DEFAULT_PATHS_CONFIG
from utils import flags

def downloading_inpaint_models(v):
    assert v in flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=DEFAULT_PATHS_CONFIG.path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(DEFAULT_PATHS_CONFIG.path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=DEFAULT_PATHS_CONFIG.path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(DEFAULT_PATHS_CONFIG.path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=DEFAULT_PATHS_CONFIG.path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(DEFAULT_PATHS_CONFIG.path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=DEFAULT_PATHS_CONFIG.path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(DEFAULT_PATHS_CONFIG.path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=DEFAULT_PATHS_CONFIG.path_loras,
        file_name=flags.PerformanceLoRA.EXTREME_SPEED.value
    )
    return flags.PerformanceLoRA.EXTREME_SPEED.value


def downloading_sdxl_lightning_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors',
        model_dir=DEFAULT_PATHS_CONFIG.path_loras,
        file_name=flags.PerformanceLoRA.LIGHTNING.value
    )
    return flags.PerformanceLoRA.LIGHTNING.value


def downloading_sdxl_hyper_sd_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        model_dir=DEFAULT_PATHS_CONFIG.path_loras,
        file_name=flags.PerformanceLoRA.HYPER_SD.value
    )
    return flags.PerformanceLoRA.HYPER_SD.value


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=DEFAULT_PATHS_CONFIG.path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(DEFAULT_PATHS_CONFIG.path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=DEFAULT_PATHS_CONFIG.path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(DEFAULT_PATHS_CONFIG.path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=DEFAULT_PATHS_CONFIG.path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(DEFAULT_PATHS_CONFIG.path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=DEFAULT_PATHS_CONFIG.path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(DEFAULT_PATHS_CONFIG.path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=DEFAULT_PATHS_CONFIG.path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(DEFAULT_PATHS_CONFIG.path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=DEFAULT_PATHS_CONFIG.path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(DEFAULT_PATHS_CONFIG.path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=DEFAULT_PATHS_CONFIG.path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(DEFAULT_PATHS_CONFIG.path_upscale_models, 'fooocus_upscaler_s409985e5.bin')

def downloading_safety_checker_model():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin',
        model_dir=DEFAULT_PATHS_CONFIG.path_safety_checker,
        file_name='stable-diffusion-safety-checker.bin'
    )
    return os.path.join(DEFAULT_PATHS_CONFIG.path_safety_checker, 'stable-diffusion-safety-checker.bin')


def download_sam_model(sam_model: str) -> str:
    match sam_model:
        case 'vit_b':
            return downloading_sam_vit_b()
        case 'vit_l':
            return downloading_sam_vit_l()
        case 'vit_h':
            return downloading_sam_vit_h()
        case _:
            raise ValueError(f"sam model {sam_model} does not exist.")


def downloading_sam_vit_b():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth',
        model_dir=DEFAULT_PATHS_CONFIG.path_sam,
        file_name='sam_vit_b_01ec64.pth'
    )
    return os.path.join(DEFAULT_PATHS_CONFIG.path_sam, 'sam_vit_b_01ec64.pth')


def downloading_sam_vit_l():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth',
        model_dir=DEFAULT_PATHS_CONFIG.path_sam,
        file_name='sam_vit_l_0b3195.pth'
    )
    return os.path.join(DEFAULT_PATHS_CONFIG.path_sam, 'sam_vit_l_0b3195.pth')


def downloading_sam_vit_h():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth',
        model_dir=DEFAULT_PATHS_CONFIG.path_sam,
        file_name='sam_vit_h_4b8939.pth'
    )
    return os.path.join(DEFAULT_PATHS_CONFIG.path_sam, 'sam_vit_h_4b8939.pth')
