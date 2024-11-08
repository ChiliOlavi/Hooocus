from ast import Dict
from typing import Optional

from pydantic import BaseModel

from ldm_patched.contrib.external import VAEDecode, EmptyLatentImage, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, \
    ControlNetApplyAdvanced
from ldm_patched.contrib.external_freelunch import FreeU_V2
from ldm_patched.contrib.external_model_advanced import ModelSamplingDiscrete, ModelSamplingContinuousEDM
from utils.config import ImageGenerationObject

class PatchSettings(BaseModel):
    sharpness: float = 2.0
    adm_scaler_end: float = 0.3
    positive_adm_scale: float = 1.5 
    negative_adm_scale: float = 0.8
    controlnet_softness: float = 0.25 
    adaptive_cfg: float = 7.0
    global_diffusion_progress: float = 0
    eps_record: Optional[float] = None


patch_settings_GLOBAL_CAUTION: Dict[int, PatchSettings] = {}
"""
Effects:
    - This dictionary stores the patch settings for each generation process
    - The key is the pid of the generation process
    - The value is an instance of PatchSettings
    
    They are set using:
    # utils/image_generation_utils.py
    def apply_patch_settings(task: ImageGenerationObject, pid: int)   
"""


opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()
opFreeU = FreeU_V2()
opModelSamplingDiscrete = ModelSamplingDiscrete()
opModelSamplingContinuousEDM = ModelSamplingContinuousEDM()


def apply_patch_settings(task: ImageGenerationObject, pid: int) -> None:
    """Apply patch settings to the global caution settings."""
    patch_settings_GLOBAL_CAUTION[pid] = PatchSettings(
        task.sample_sharpness,
        task.adm_scaler_end,
        task.adm_scaler_positive,
        task.adm_scaler_negative,
        task.controlnet_softness,
        task.adaptive_cfg,
    )
