import sys, os
from unittest.mock import Base

from pydantic import BaseModel, Field

from modules.extra_utils import get_files_from_folder
from modules.model_loader import load_file_from_url
from utils.hooocus_utils import LoraTuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enum import IntEnum, Enum
from modules import style_sorter
from utils import config
from typing import List, Literal, Optional

from utils.consts import DEFAULT_PATHS_CONFIG

from modules.sdxl_styles import legal_style_names


enhancement_uov_before = "Before First Enhancement"
enhancement_uov_after = "After Last Enhancement"
enhancement_uov_processing_order = [enhancement_uov_before, enhancement_uov_after]
LITERAL_ENHANCEMENT_UOV_PROCESSING_ORDER = Literal["Before First Enhancement", "After Last Enhancement"]

enhancement_uov_prompt_type_original = 'Original Prompts'
enhancement_uov_prompt_type_last_filled = 'Last Filled Enhancement Prompts'
enhancement_uov_prompt_types = [enhancement_uov_prompt_type_original, enhancement_uov_prompt_type_last_filled]

CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

# fooocus: a1111 (Civitai)
KSAMPLER = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "",
    "dpmpp_3m_sde_gpu": "",
    "ddpm": "",
    "lcm": "LCM",
    "tcd": "TCD",
    "restart": "Restart"
}

SAMPLER_EXTRA = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": ""
}

SAMPLERS = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES = list(KSAMPLER.keys())

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "lcm", "turbo", "align_your_steps", "tcd", "edm_playground_v2.5"]
SAMPLER_NAMES = KSAMPLER_NAMES + list(SAMPLER_EXTRA.keys())

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES

clip_skip_max = 12

default_vae = 'Default (model)'

refiner_swap_method = 'joint'

default_input_image_tab = 'uov_tab'





class OutputFormat(Enum):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: c.value, cls))


class PerformanceLoRA(Enum):
    QUALITY = None
    SPEED = None
    EXTREME_SPEED = 'sdxl_lcm_lora.safetensors'
    LIGHTNING = 'sdxl_lightning_4step_lora.safetensors'
    HYPER_SD = 'sdxl_hyper_sd_4step_lora.safetensors'


class Steps(IntEnum):
    QUALITY = 60
    SPEED = 30
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4

    @classmethod
    def keys(cls) -> list:
        return list(map(lambda c: c, Steps.__members__))


class StepsUOV(IntEnum):
    QUALITY = 36
    SPEED = 18
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4


class Performance(Enum):
    QUALITY = 'Quality'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    LIGHTNING = 'Lightning'
    HYPER_SD = 'Hyper-SD'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: (c.name, c.value), cls))

    @classmethod
    def values(cls) -> list:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def by_steps(cls, steps: int | str):
        return cls[Steps(int(steps)).name]

    @classmethod
    def has_restricted_features(cls, x) -> bool:
        if isinstance(x, Performance):
            x = x.value
        return x in [cls.EXTREME_SPEED.value, cls.LIGHTNING.value, cls.HYPER_SD.value]

    def steps(self) -> int | None:
        return Steps[self.name].value if self.name in Steps.__members__ else None

    def steps_uov(self) -> int | None:
        return StepsUOV[self.name].value if self.name in StepsUOV.__members__ else None

    def lora_filename(self) -> str | None:
        return PerformanceLoRA[self.name].value if self.name in PerformanceLoRA.__members__ else None



class _BaseModelFile(BaseModel):
    model_path_basename: str
    model_name: str
    model_url: str = None
    model_path_folder: str = None

    def download_model(self):
        if not self.model_path_folder:
            raise ValueError("model_path_folder is not set.")

        if self.model_url is None:
            if self.model_path_basename:
                return os.path.join(self.model_path_folder, self.model_path_basename)
        load_file_from_url(
            url=self.model_url,
            model_dir=self.model_path_folder,
            file_name=self.model_name
        )
        return os.path.join(self.model_path_folder, self.model_path_basename)

class BaseControlNetModelFiles(Enum):
    class _BaseControlNetModelFile(_BaseModelFile):
        model_path_folder = DEFAULT_PATHS_CONFIG.path_controlnet

    
    ImagePromptClipVIsion = _BaseControlNetModelFile(
        model_name = "clip_vision_vit_h",
        model_url = "https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        model_path_basename = "clip_vision_vit_h.safetensors",

    )

    ImagePromptAdapterPlus = _BaseControlNetModelFile(
        model_name = "ip-adapter-plus",
        model_url = "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
        model_path_basename = "'ip-adapter-plus_sdxl_vit-h.bin"
    )

    ImagePromptAdapterNegative = _BaseControlNetModelFile(
        model_name = "fooocus_ip_negative",
        model_url = "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        model_path_basename = "fooocus_ip_negative.safetensors"
    )

    ImagePromptAdapterFace = _BaseControlNetModelFile(
        model_name = "ip-adapter-plus-face",
        model_url = "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
        model_path_basename = "ip-adapter-plus-face_sdxl_vit-h.bin"
    )

    PyraCanny = _BaseControlNetModelFile(
        model_name = 'canny',
        model_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_path_basename = 'control-lora-canny-rank128.safetensors',
    )

    CPDS = _BaseControlNetModelFile(
        model_name = 'cpds',
        model_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_path_basename = 'fooocus_xl_cpds_128.safetensors',
    )


class InpaintModelFiles(Enum):
    class _InpaintModelFile(_BaseModelFile):
        model_path_folder = DEFAULT_PATHS_CONFIG.path_inpaint

    InpaintHead = _InpaintModelFile(
        model_name = 'fooocus_inpaint_head.pth',
        model_url = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_path_basename = 'fooocus_inpaint_head.pth'
    )

    InpaintPatchV1 = _InpaintModelFile(
        model_name = 'inpaint.fooocus.patch',
        model_url = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
        model_path_basename = 'inpaint.fooocus.patch'
    )

    InpaintPatchV25 = _InpaintModelFile(
        model_name = 'inpaint_v25.fooocus.patch',
        model_url = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
        model_path_basename = 'inpaint_v25.fooocus.patch'
    )

    InpaintPatchV26 = _InpaintModelFile(
        model_name = 'inpaint_v26.fooocus.patch',
        model_url = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
        model_path_basename = 'inpaint_v26.fooocus.patch'
    )

class SAM_Files(Enum):
    """
    Segment Anything Model Files

    """
    class _SAMFile(_BaseModelFile):
        model_path_folder = DEFAULT_PATHS_CONFIG.path_sam

    VIT_B = _SAMFile(
        model_name = 'sam_vit_b_01ec64.pth',
        model_url = 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth',
        model_path_basename = 'sam_vit_b_01ec64.pth'
    )

    VIT_L = _SAMFile(
        model_name = 'sam_vit_l_0b3195.pth',
        model_url = 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth',
        model_path_basename = 'sam_vit_l_0b3195.pth'
    )

    VIT_H = _SAMFile(
        model_name = 'sam_vit_h_4b8939.pth',
        model_url = 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth',
        model_path_basename = 'sam_vit_h_4b8939.pth'
    )


    


class _BaseControlNetTask(BaseModel):
    stop: float = Field(0.5, ge=0, le=1)
    weight: float = Field(1.0, ge=0, le=1)
    models: Optional[List[BaseControlNetModelFiles]] = None

class ControlNetTasks(Enum):
    ImagePrompt = _BaseControlNetTask(
        stop = 0.5,
        weight = 0.6,
        models = [
            BaseControlNetModelFiles.ImagePromptClipVIsion, 
            BaseControlNetModelFiles.ImagePromptAdapterPlus,
            BaseControlNetModelFiles.ImagePromptAdapterNegative
        ]
    )
    
    FaceSwap = _BaseControlNetTask(
        stop = 0.9,
        weight = 0.75,
        models = [
            BaseControlNetModelFiles.ImagePromptClipVIsion,
            BaseControlNetModelFiles.ImagePromptAdapterFace,
            BaseControlNetModelFiles.ImagePromptAdapterNegative
        ]
    )

    PyraCanny = _BaseControlNetTask(
        stop = 0.5,
        weight = 1.0,
        models = [
            BaseControlNetModelFiles.PyraCanny
        ]
        )

    CPDS = _BaseControlNetTask(    
        stop = 0.5,
        weight = 1.0,
        models = [
            BaseControlNetModelFiles.CPDS
        ]
    )


UpscaleModel = _BaseModelFile(
    model_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
    model_name="fooocus_upscaler",
    model_path_basename="fooocus_upscaler_s409985e5.bin",
    model_path_folder=DEFAULT_PATHS_CONFIG.path_upscale_models
)

SafetyCheckModel = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
    model_name="fooocus_safety_check",
    model_path_basename="stable-diffusion-safety-checker.bin",
    model_path_folder=DEFAULT_PATHS_CONFIG.path_safety_checker
)

SDXL_LightningLoRA = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
    model_name=PerformanceLoRA.LIGHTNING.value,
    model_path_basename=PerformanceLoRA.LIGHTNING.value,
    model_path_folder=DEFAULT_PATHS_CONFIG.path_loras
)

SDXL_HyperSDLoRA = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
    model_name=PerformanceLoRA.HYPER_SD.value,
    model_path_basename=PerformanceLoRA.HYPER_SD.value,
    model_path_folder=DEFAULT_PATHS_CONFIG.path_loras
)

SDXL_LCM_LoRA = _BaseModelFile(
    model_url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
    model_name=PerformanceLoRA.EXTREME_SPEED.value,
    model_path_basename=PerformanceLoRA.EXTREME_SPEED.value,
    model_path_folder=DEFAULT_PATHS_CONFIG.path_loras
)


output_formats = ['png', 'jpeg', 'webp']
LITERAL_OUTPUT_FORMATS = Literal['png', 'jpeg', 'webp']

inpaint_mask_models = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime', 'sam']
inpaint_mask_cloth_category = ['full', 'upper', 'lower']
inpaint_mask_sam_model = ['vit_b', 'vit_l', 'vit_h']

inpaint_engine_versions = ['None', 'v1', 'v2.5', 'v2.6']
inpaint_option_default = 'Inpaint or Outpaint (default)'
inpaint_option_detail = 'Improve Detail (face, hand, eyes, etc.)'
inpaint_option_modify = 'Modify Content (add objects, change background, etc.)'
inpaint_options = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]

describe_type_photo = 'Photograph'
describe_type_anime = 'Art/Anime'
describe_types = [describe_type_photo, describe_type_anime]

sdxl_aspect_ratios = [
    '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
    '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
    '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
    '1664*576', '1728*576'
]

example_enhance_detection_prompts = ['face', 'eye', 'mouth', 'hair', 'hand', 'body']
example_inpaint_prompts = [
        'highly detailed face', 
        'detailed girl face', 
        'detailed man face', 
        'detailed hand', 
        'beautiful eyes']



model_filenames = []
lora_filenames = []
vae_filenames = []
wildcard_filenames = []

def get_model_filenames(folder_paths, extensions=None, name_filter=None):
    if extensions is None:
        extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']
    files = []

    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder in folder_paths:
        files += get_files_from_folder(folder, extensions, name_filter)

    return files


model_filenames = get_model_filenames(DEFAULT_PATHS_CONFIG.path_checkpoints)
lora_filenames = get_model_filenames(DEFAULT_PATHS_CONFIG.path_loras)
vae_filenames = get_model_filenames(DEFAULT_PATHS_CONFIG.path_vae)
wildcard_filenames = get_files_from_folder(DEFAULT_PATHS_CONFIG.path_wildcards, ['.txt'])

def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    return presets + [f[:f.index(".json")] for f in os.listdir(preset_folder) if f.endswith('.json')]

available_presets = get_presets()




style_sorter.try_load_sorted_styles(legal_style_names, config.default_styles)
all_styles = style_sorter.all_styles
all_loras = config.lora_filenames
performance_lora_keys = PerformanceLoRA.__members__.keys()
performance_keys = Performance.__members__.keys()

ALLOWED_TABS = Literal['uov', 'inpaint', 'ip', 'desc', 'enhance', 'metadata']
OUTPAINT_SELECTIONS = Literal['Left', 'Right', 'Top', 'Bottom']
REFINER_SWAP_METHODS = Literal['joint', 'separate', 'vae']
CONTROLNET_TASK_TYPES = Literal["ImagePrompt", "FaceSwap", "PyraCanny", "CPDS"]    

UPSCALE_OR_VARIATION_MODES = Literal[
        'Enabled',
        'Vary (Subtle)',
        'Vary (Strong)',
        'Upscale (1.5x)',
        'Upscale (2x)',
        'Upscale (Fast 2x)',
]

