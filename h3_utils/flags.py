import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy
from pydantic import BaseModel, Field

from h3_utils.filesystem_utils import get_files_from_folder, get_model_filenames, get_presets
from h3_utils.path_configs import FolderPathsConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from enum import IntEnum, Enum
from typing import Literal
import tempfile
from h3_utils.logging_util import LoggingUtil

log = LoggingUtil().get_logger()


random_style_name = 'Random Style'

DESCRIBE_TYPE_PHOTO = 'Photograph'
DESCRIBE_TYPE_ANIME = 'Art/Anime'
DESCRIBE_TYPES = [DESCRIBE_TYPE_PHOTO, DESCRIBE_TYPE_ANIME]

literal_enhancement_proceccing_order = Literal["Before First Enhancement", "After Last Enhancement"]

ENHANCEMENT_UOV_BEFORE = "Before First Enhancement"
ENHANCEMENT_UOV_AFTER = "After Last Enhancement"
ENHANCEMENT_UOV_PROCESSING_ORDER = [ENHANCEMENT_UOV_BEFORE, ENHANCEMENT_UOV_AFTER]

ENHANCEMENT_UOV_PROMPT_TYPE_ORIGINAL = 'Original Prompts'
ENHANCEMENT_UOV_PROMPT_TYPE_LAST_FILLED = 'Last Filled Enhancement Prompts'
ENHANCEMENT_UOV_PROMPT_TYPES = [ENHANCEMENT_UOV_PROMPT_TYPE_ORIGINAL, ENHANCEMENT_UOV_PROMPT_TYPE_LAST_FILLED]

SDXL_ASPECT_RATIOS = Literal[
    '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
    '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
    '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
    '1664*576', '1728*576'
]

INPUT_IMAGE_MODES = Literal['uov', 'inpaint', 'ip', 'desc', 'enhance', 'metadata']
class INPUT_IMAGE_MODES_CLASS:
    uov = 'uov'
    ip = 'ip'
    inpaint = 'inpaint'
    desc = 'desc'
    enhance = 'enhance'
    metadata = 'metadata'


class Overrides(BaseModel):
    steps: int | None = None
    switch: float | None = None
    width: int | None = None
    height: int | None = None

OUTPAINT_SELECTIONS = Literal['Left', 'Right', 'Top', 'Bottom']

REFINER_SWAP_METHODS = Literal['joint', 'separate', 'vae']

CONTROLNET_TASK_TYPES = Literal["ImagePrompt", "FaceSwap", "PyraCanny", "CPDS"]    

EXAMPLE_ENHANCE_DETECTION_PROMPTS = [
        'face', 'eye', 'mouth', 'hair', 'hand', 'body'
    ],

UPSCALE_OR_VARIATION_MODES = Literal[
        'Enabled',
        'Vary (Subtle)',
        'Vary (Strong)',
        'Upscale (1.5x)',
        'Upscale (2x)',
        'Upscale (Fast 2x)',
]

CIVITAI_NO_KARRAS = Literal["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

max_image_number: int = 32
max_lora_number: int = 5
loras_max_weight: float = 2.0
loras_min_weight: float = 2.0
sam_max_detections: int = 0 #The default sam max detections to use.

INPAINT_MASK_CLOTH_CATEGORY = Literal['full', 'upper', 'lower']

class KSAMPLER(Enum): 
    euler = "Euler"
    euler_ancestral = "Euler a"
    heun = "Heun"
    heunpp2 = ""
    dpm_2 = "DPM2"
    dpm_2_ancestral = "DPM2 a"
    lms = "LMS"
    dpm_fast = "DPM fast"
    dpm_adaptive = "DPM adaptive"
    dpmpp_2s_ancestral = "DPM++ 2S a"
    dpmpp_sde = "DPM++ SDE"
    dpmpp_sde_gpu = "DPM++ SDE"
    dpmpp_2m = "DPM++ 2M"
    dpmpp_2m_sde = "DPM++ 2M SDE"
    dpmpp_2m_sde_gpu = "DPM++ 2M SDE"
    dpmpp_3m_sde = ""
    dpmpp_3m_sde_gpu = ""
    ddpm = ""
    lcm = "LCM"
    tcd = "TCD"
    restart = "Restart"

class EXTRA_KSAMPLER(Enum):
    ddim = "DDIM"
    uni_pc = "UniPC"
    uni_pc_bh2 = ""

# Both KSAMPLER and EXTRA_KSAMPLER
KSAMPLER_NAMES = [k.value for k in KSAMPLER] + [k.value for k in EXTRA_KSAMPLER]
SAMPLERS = KSAMPLER | EXTRA_KSAMPLER
DEFAULT_SAMPLER = KSAMPLER.dpmpp_2m_sde_gpu

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "lcm", "turbo", "align_your_steps", "tcd", "edm_playground_v2.5"]
SCHEDULER_NAMES_LITERAL = Literal[SCHEDULER_NAMES]

class _AvailableConfigsBase(Enum):
    pass


class LatentPreviewMethod(_AvailableConfigsBase):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "fast"
    TAESD = "taesd"


class OutputFormat(Enum):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'
    
class PerformanceLoRA(_AvailableConfigsBase):
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

class Performance(_AvailableConfigsBase):
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




INPAINT_ENGINE_VERSIONS = Literal["1.0", "2.5", "2.6"]
AVAILABLE_PRESETS = get_presets()
MODEL_FILENAMES = get_model_filenames(FolderPathsConfig.path_checkpoints)
LORA_FILENAMES = get_model_filenames(FolderPathsConfig.path_loras)
VAE_FILENAMES = get_model_filenames(FolderPathsConfig.path_vae)
WILDCARD_FILENAMES = get_files_from_folder(FolderPathsConfig.path_wildcards, ['.txt'])

performance_lora_keys = PerformanceLoRA.__members__.keys()
performance_keys = Performance.__members__.keys()




