import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from pydantic import BaseModel

from modules.model_loader import load_file_from_url
from utils.config import PathsConfig
from utils.file_sort_utils import get_files_from_folder, get_model_filenames
from utils.flags import PerformanceLoRA



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
        model_path_folder = PathsConfig.path_controlnet.value
    
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
        model_path_folder = PathsConfig.path_inpaint.value

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
        model_path_folder = PathsConfig.path_sam.value

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
    model_path_folder=PathsConfig.path_upscale_models.value
)

SafetyCheckModel = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
    model_name="fooocus_safety_check",
    model_path_basename="stable-diffusion-safety-checker.bin",
    model_path_folder=PathsConfig.path_safety_checker.value
)

SDXL_LightningLoRA = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
    model_name=PerformanceLoRA.LIGHTNING.value,
    model_path_basename=PerformanceLoRA.LIGHTNING.value,
    model_path_folder=PathsConfig.path_loras.value
)

SDXL_HyperSDLoRA = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
    model_name=PerformanceLoRA.HYPER_SD.value,
    model_path_basename=PerformanceLoRA.HYPER_SD.value,
    model_path_folder=PathsConfig.path_loras.value
)

SDXL_LCM_LoRA = _BaseModelFile(
    model_url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
    model_name=PerformanceLoRA.EXTREME_SPEED.value,
    model_path_basename=PerformanceLoRA.EXTREME_SPEED.value,
    model_path_folder=PathsConfig.path_loras.value
)


MODEL_FILENAMES = get_model_filenames(PathsConfig.path_checkpoints.value)
LORA_FILENAMES = get_model_filenames(PathsConfig.path_loras.value)
VAE_FILENAMES = get_model_filenames(PathsConfig.path_vae.value)
WILDCARD_FILENAMES = get_files_from_folder(PathsConfig.path_wildcards.value, ['.txt'])