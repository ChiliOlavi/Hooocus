import os
from enum import Enum
from typing import List, Optional
import numpy
from pydantic import BaseModel, Field

from pydantic import BaseModel

from modules.model_loader import load_file_from_url
from utils.path_configs import FolderPathsConfig
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

class BaseControlNetModelFiles:
    class _BaseControlNetModelFile(_BaseModelFile):
        model_path_folder = FolderPathsConfig.path_controlnet.value
        def full_path(self):
            return os.path.join(self.model_path_folder, self.model_path_basename)
    
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
        controlnet_name = 'canny',
        model_name = 'canny',
        model_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_path_basename = 'control-lora-canny-rank128.safetensors',
    )

    CPDS = _BaseControlNetModelFile(
        controlnet_name = 'cpds',
        model_name = 'cpds',
        model_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_path_basename = 'fooocus_xl_cpds_128.safetensors',
    )

class InpaintModelFiles:
    class _InpaintModelFile(_BaseModelFile):
        model_path_folder = FolderPathsConfig.path_inpaint.value

    def download_based_on_version(self, version):
        if version == "1.0":
            return self.InpaintPatchV1.download_model()
        elif version == "2.5":
            return self.InpaintPatchV25.download_model()
        elif version == "2.6":
            return self.InpaintPatchV26.download_model()
        else:
            raise ValueError("Invalid version number.")


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
        model_path_folder = FolderPathsConfig.path_sam.value

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


    
class BaseControlNetTask(BaseModel):
    ip_conds: Optional[List[Any]] = None
    ip_unconds: Optional[List[Any]] = None
    stop: float = Field(0.5, ge=0, le=1)
    img: Optional[numpy.ndarray] = None
    weight: float = Field(1.0, ge=0, le=1)
    models: Optional[List[BaseControlNetModelFiles._BaseControlNetModelFile]] = None
    name: str = Field(None, description="Name of the ControlNetTask.")
    model_paths: Optional[List[str]] = None

    def get_paths(self):
        if self.models is None:
            return []
        return [model.full_path() for model in self.models]

class ControlNetTasks(BaseModel):
    ImagePrompt: BaseControlNetTask = BaseControlNetTask(
        stop = 0.5,
        name = "ImagePrompt",
        weight = 0.6,
        img = None,
        models = [
            BaseControlNetModelFiles.ImagePromptClipVIsion, 
            BaseControlNetModelFiles.ImagePromptAdapterPlus,
            BaseControlNetModelFiles.ImagePromptAdapterNegative
        ]
    )
    
    FaceSwap: BaseControlNetTask = BaseControlNetTask(
        stop = 0.9,
        img = None,
        name = "FaceSwap",
        weight = 0.75,
        models = [
            BaseControlNetModelFiles.ImagePromptClipVIsion,
            BaseControlNetModelFiles.ImagePromptAdapterFace,
            BaseControlNetModelFiles.ImagePromptAdapterNegative
        ],
    )

    PyraCanny: BaseControlNetTask = BaseControlNetTask(
        stop = 0.5,
        img = None,
        name = "PyraCanny",
        weight = 1.0,
        models = [
            BaseControlNetModelFiles.PyraCanny
        ]

        )

    CPDS: BaseControlNetTask = BaseControlNetTask(
        stop = 0.5,
        img = None,
        name = "CPDS",
        weight = 1.0,
        models = [
            BaseControlNetModelFiles.CPDS
        ]
    )

UpscaleModel = _BaseModelFile(
    model_url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
    model_name="fooocus_upscaler",
    model_path_basename="fooocus_upscaler_s409985e5.bin",
    model_path_folder=FolderPathsConfig.path_upscale_models.value
)

SafetyCheckModel = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
    model_name="fooocus_safety_check",
    model_path_basename="stable-diffusion-safety-checker.bin",
    model_path_folder=FolderPathsConfig.path_safety_checker.value
)

SDXL_LightningLoRA = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
    model_name=PerformanceLoRA.LIGHTNING.value,
    model_path_basename=PerformanceLoRA.LIGHTNING.value,
    model_path_folder=FolderPathsConfig.path_loras.value
)

SDXL_HyperSDLoRA = _BaseModelFile(
    model_url="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
    model_name=PerformanceLoRA.HYPER_SD.value,
    model_path_basename=PerformanceLoRA.HYPER_SD.value,
    model_path_folder=FolderPathsConfig.path_loras.value
)

SDXL_LCM_LoRA = _BaseModelFile(
    model_url="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
    model_name=PerformanceLoRA.EXTREME_SPEED.value,
    model_path_basename=PerformanceLoRA.EXTREME_SPEED.value,
    model_path_folder=FolderPathsConfig.path_loras.value
)


