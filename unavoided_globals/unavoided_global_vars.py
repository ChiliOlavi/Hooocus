from typing import Optional, Dict

from pydantic import BaseModel

from ldm_patched.contrib.external import VAEDecode, EmptyLatentImage, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, \
    ControlNetApplyAdvanced
from ldm_patched.contrib.external_freelunch import FreeU_V2
from ldm_patched.contrib.external_model_advanced import ModelSamplingDiscrete, ModelSamplingContinuousEDM
from h3_utils.config import ImageGenerationObject

from h3_utils.logging_util import LoggingUtil
log = LoggingUtil("GlobalVarManager").get_logger()

"""
This is of course a testament to the fact that the codebase is not that well-structured yet.

All of the known globals are defined here, and they are used in various parts of the codebase.

I'll try to tag the places where they are used with "# GLOBAL VAR USAGE"

"""

class PatchSettings(BaseModel):
    class Config:
        orm_mode = True

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

inpaintworker_current_task_GLOBAL_CAUTION = None

def apply_patch_settings(pid: int, task: ImageGenerationObject) -> None:
    """Apply patch settings to the global caution settings."""
    log.warning(f"Applying patch settings for pid {pid}")
    patch_settings_GLOBAL_CAUTION[pid] = PatchSettings(
        sample_sharpness = task.sample_sharpness,
        adm_scaler_end = task.adm_scaler_end,
        adm_scaler_positive = task.adm_scaler_positive,
        adm_scaler_negative = task.adm_scaler_negative,
        controlnet_softness = task.controlnet_softness,
        adaptive_cfg = task.adaptive_cfg,
    )


