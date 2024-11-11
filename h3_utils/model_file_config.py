import os
import sys
import numpy

ROOT_DIR = os.path.abspath(__file__).split("h3_utils")[0]
sys.path.append(ROOT_DIR)


from enum import Enum

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from modules.model_file_utils.model_loader import load_file_from_url
from h3_utils.path_configs import FolderPathsConfig
from h3_utils.flags import PerformanceLoRA


class _BaseModelFile(BaseModel):
    """A base class for model files

    Attributes:
        - model_path_basename (str) The name of the model file.
        - model_path_folder (str) The folder path where the model will be stored.
        - model_url (str) The URL to download the model file.
        - nameof_model (str) The name of the model file.

    Methods:
        - download_model() -> str: Downloads the model file and returns the full path.
    """
    basename_of_model: str = None
    folder_path_of_model: str = FolderPathsConfig.path_controlnet
    name_of_model: str = None
    url_of_model: str = None
    
    def full_path(self):
        return os.path.join(self.model_path_folder, self.model_path_basename)

    def download_model(self):
        if not self.model_path_folder:
            raise ValueError("model_path_folder is not set.")

        if self.nameof_model is None:
            if self.model_path_basename:
                return os.path.join(self.model_path_folder, self.model_path_basename)
        load_file_from_url(
            url=self.model_url,
            model_dir=self.model_path_folder,
            file_name=self.nameof_model
        )
        return os.path.join(self.model_path_folder, self.model_path_basename)



class _BaseControlNetModelFile(_BaseModelFile):
    folder_path_of_model: str = FolderPathsConfig.path_controlnet
    def full_path(self):
        return os.path.join(self.model_path_folder, self.model_path_basename)

ImagePromptClipVIsion = _BaseControlNetModelFile(
    name_of_model = "clip_vision_vit_h",
    url_of_model = "https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
    basename_of_model = "clip_vision_vit_h.safetensors",
)

ImagePromptAdapterPlus = _BaseControlNetModelFile(
    name_of_model = "ip-adapter-plus",
    url_of_model = "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
    basename_of_model = "'ip-adapter-plus_sdxl_vit-h.bin"
)

ImagePromptAdapterNegative = _BaseControlNetModelFile(
    name_of_model = "fooocus_ip_negative",
    url_of_model = "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
    basename_of_model = "fooocus_ip_negative.safetensors"
)

ImagePromptAdapterFace = _BaseControlNetModelFile(
    name_of_model = "ip-adapter-plus-face",
    url_of_model = "https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
    basename_of_model = "ip-adapter-plus-face_sdxl_vit-h.bin"
)

PyraCanny = _BaseControlNetModelFile(
    name_of_model = 'canny',
    url_of_model = 'https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
    basename_of_model = 'control-lora-canny-rank128.safetensors',
)

CPDS = _BaseControlNetModelFile(
    name_of_model = 'cpds',
    url_of_model = 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
    basename_of_model = 'fooocus_xl_cpds_128.safetensors',
)

class InpaintModelFiles:
    class _InpaintModelFile(_BaseModelFile):
        folder_path_of_model: str = FolderPathsConfig.path_inpaint

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
        name_of_model = 'fooocus_inpaint_head.pth',
        url_of_model = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        basename_of_model = 'fooocus_inpaint_head.pth'
    )

    InpaintPatchV1 = _InpaintModelFile(
        name_of_model = 'inpaint.fooocus.patch',
        url_of_model = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
        basename_of_model = 'inpaint.fooocus.patch'
    )

    InpaintPatchV25 = _InpaintModelFile(
        name_of_model = 'inpaint_v25.fooocus.patch',
        url_of_model = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
        basename_of_model = 'inpaint_v25.fooocus.patch'
    )

    InpaintPatchV26 = _InpaintModelFile(
        name_of_model = 'inpaint_v26.fooocus.patch',
        url_of_model = 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
        basename_of_model = 'inpaint_v26.fooocus.patch'
    )

class _SAMFile(_BaseModelFile):
    folder_path_of_model: str = FolderPathsConfig.path_sam

class SAM_Files(Enum):
    """
    Segment Anything Model Files

    """

    VIT_B = _SAMFile(
        name_of_model = 'sam_vit_b_01ec64.pth',
        url_of_model = 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth',
        basename_of_model = 'sam_vit_b_01ec64.pth'
    )

    VIT_L = _SAMFile(
        name_of_model = 'sam_vit_l_0b3195.pth',
        url_of_model = 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth',
        basename_of_model = 'sam_vit_l_0b3195.pth'
    )

    VIT_H = _SAMFile(
        name_of_model = 'sam_vit_h_4b8939.pth',
        url_of_model = 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth',
        basename_of_model = 'sam_vit_h_4b8939.pth'
    )



class VaeApproxFiles:
    VaeAppSDXL = _BaseModelFile(
        folder_path_of_model = FolderPathsConfig.path_vae,
        name_of_model = 'xlvaeapp.pth',
        url_of_model = 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth',
        basename_of_model = 'xlvaeapp.pth'
    )

    VaeAppSD15 = _BaseModelFile(
        folder_path_of_model= FolderPathsConfig.path_vae,
        name_of_model = 'vaeapp_sd15.pth',
        url_of_model = 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt',
        basename_of_model = 'vaeapp_sd15.pth'
    )

    XlToV1Interposer = _BaseModelFile(
        folder_path_of_model = FolderPathsConfig.path_vae,
        name_of_model = 'xl-to-v1_interposer-v4.0.safetensors',
        url_of_model = 'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors',
        basename_of_model = 'xl-to-v1_interposer-v4.0.safetensors'
    )


    
class BaseControlNetTask(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    ip_conds: Optional[List[Any]] = None
    ip_unconds: Optional[List[Any]] = None
    stop: float = Field(0.5, ge=0, le=1)
    img: Optional[numpy.ndarray] = None
    weight: float = Field(1.0, ge=0, le=1)
    all_models: Optional[List[_BaseControlNetModelFile]] = None
    name: str = Field(None, description="Name of the ControlNetTask.")
    paths_of_models: Optional[List[str]] = None

    def get_paths(self):
        if self.models is None:
            return []
        return [model.full_path() for model in self.models]

class ControlNetTasks:
    ImagePrompt: BaseControlNetTask = BaseControlNetTask(
        stop = 0.5,
        name = "ImagePrompt",
        weight = 0.6,
        img = None,
        all_models = [
            ImagePromptClipVIsion, 
            ImagePromptAdapterPlus,
            ImagePromptAdapterNegative
        ]
    )
    
    FaceSwap: BaseControlNetTask = BaseControlNetTask(
        stop = 0.9,
        img = None,
        name = "FaceSwap",
        weight = 0.75,
        all_models = [
            ImagePromptClipVIsion,
            ImagePromptAdapterFace,
            ImagePromptAdapterNegative
        ],
    )

    PyraCanny: BaseControlNetTask = BaseControlNetTask(
        stop = 0.5,
        img = None,
        name = "PyraCanny",
        weight = 1.0,
        all_models = [
            PyraCanny
        ]

        )

    CPDS: BaseControlNetTask = BaseControlNetTask(
        stop = 0.5,
        img = None,
        name = "CPDS",
        weight = 1.0,
        all_models = [
            CPDS
        ]
    )

UpscaleModel = _BaseModelFile(
    url_of_model="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
    name_of_model="fooocus_upscaler",
    basename_of_model="fooocus_upscaler_s409985e5.bin",
    folder_path_of_model=FolderPathsConfig.path_upscale_models
)

SafetyCheckModel = _BaseModelFile(
    url_of_model="https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin",
    name_of_model="fooocus_safety_check",
    basename_of_model="stable-diffusion-safety-checker.bin",
    folder_path_of_model=FolderPathsConfig.path_safety_checker
)

SDXL_LightningLoRA = _BaseModelFile(
    url_of_model="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors",
    name_of_model=PerformanceLoRA.LIGHTNING.value,
    basename_of_model=PerformanceLoRA.LIGHTNING.value,
    folder_path_of_model=FolderPathsConfig.path_loras
)

SDXL_HyperSDLoRA = _BaseModelFile(
    url_of_model="https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors",
    name_of_model=PerformanceLoRA.HYPER_SD.value,
    basename_of_model=PerformanceLoRA.HYPER_SD.value,
    folder_path_of_model=FolderPathsConfig.path_loras
)

SDXL_LCM_LoRA = _BaseModelFile(
    url_of_model="https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors",
    name_of_model=PerformanceLoRA.EXTREME_SPEED.value,
    basename_of_model=PerformanceLoRA.EXTREME_SPEED.value,
    folder_path_of_model=FolderPathsConfig.path_loras
)


class AllModelFiles:

    BaseModel = _BaseModelFile()
    UpscaleModel = UpscaleModel
    SafetyCheckModel = SafetyCheckModel
    SDXL_LightningLoRA = SDXL_LightningLoRA
    SDXL_HyperSDLoRA = SDXL_HyperSDLoRA
    SDXL_LCM_LoRA = SDXL_LCM_LoRA
    ControlNetModels = [ControlNetTasks.ImagePrompt.all_models, ControlNetTasks.FaceSwap.all_models, ControlNetTasks.PyraCanny.all_models, ControlNetTasks.CPDS.all_models]
    InpaintModels = InpaintModelFiles()
    SAM_Files = SAM_Files
    

