import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from enum import IntEnum, Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from modules.extra_utils import get_files_from_folder
from modules.model_loader import load_file_from_url
from modules import style_sorter
from modules.sdxl_styles import legal_style_names

from utils import config



class AvailableConfigs:
    enhancement_uov_before = "Before First Enhancement"
    enhancement_uov_after = "After Last Enhancement"
    enhancement_uov_processing_order = [enhancement_uov_before, enhancement_uov_after]
    literal_enhancement_proceccing_order = Literal["Before First Enhancement", "After Last Enhancement"]
    enhancement_uov_prompt_type_original = 'Original Prompts'
    enhancement_uov_prompt_type_last_filled = 'Last Filled Enhancement Prompts'
    enhancement_uov_prompt_types = [enhancement_uov_prompt_type_original, enhancement_uov_prompt_type_last_filled]

CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]


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
KSAMPLER_NAMES_LITERAL = Literal[KSAMPLER_NAMES]
SAMPLERS = KSAMPLER | EXTRA_KSAMPLER

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "lcm", "turbo", "align_your_steps", "tcd", "edm_playground_v2.5"]

class _AvailableConfigsBase(Enum):

    @classmethod
    def literalize(cls):
        #values_held = [v for k, v in cls.__dict__.items() if not k.startswith('_') and not callable(v) and not isinstance(v, classmethod)]
        values_held = [v.value for v in cls]
        return Literal[tuple(values_held)]
    

class OutputFormat(_AvailableConfigsBase):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'
    
OUTPUT_FORMATS = OutputFormat.literalize()

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


def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    return presets + [f[:f.index(".json")] for f in os.listdir(preset_folder) if f.endswith('.json')]

AVAILABLE_PRESETS = get_presets()


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

