from ast import List
from email.mime import image
import os, sys

from utils import flags
from utils.data_models import PatchSettings
current_dir = os.path.dirname(os.path.abspath(__file__))
running_dir = os.path.dirname(current_dir)
sys.path.append(running_dir)
# This is a workaround to import modules from the parent directory

import numpy
import random
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

from utils import config
from utils.flags import CONTROLNET_TASK_TYPES
from utils.config import GLOBAL_CONFIG
import modules.style_sorter as style_sorter


class LoraTuple(BaseModel):
    enabled: bool = False
    model: str = ""
    weight: float = 1.0

    @field_validator("weight", mode="after")
    def validate_lora_weight(cls, v):
        if v < GLOBAL_CONFIG.default_loras_min_weight or v > GLOBAL_CONFIG.default_loras_max_weight:
            raise ValueError(f"Invalid lora_weight value: {v}")
        return v
    
    def to_array(self):
        if self.enabled:
            return [self.model, self.weight] 




class ControlNetImageTask(BaseModel):
    cn_img: Optional[numpy.ndarray] = None
    cn_stop: Optional[float] = None
    cn_weight: Optional[float] = 1.0
    cn_type: Optional[str] = flags.default_ip

    class Config:
        arbitrary_types_allowed = True

initial_cn_tasks = {x: [] for x in flags.ip_list}
for initial_cn_task in initial_cn_tasks: # Hoping that the controlnet image count is 4
    initial_cn_tasks[initial_cn_task] = ControlNetImageTask(cn_img=None, cn_stop=flags.default_parameters[initial_cn_task][0], cn_weight=flags.default_parameters[initial_cn_task][1])

class EnhanceMaskCtrls(BaseModel):
    """
    Enhacement mask controls for inpaint and outpaint
    
    """
    enhance_mask_dino_prompt_text: str = config.example_enhance_detection_prompts[0]
    enhance_prompt: str = None # Enhacement positive prompt. Uses original prompt if None
    enhance_negative_prompt: str = None # Enhacement negative prompt. Uses original negative prompt if None
    enhance_mask_model: str = config.default_enhance_inpaint_mask_model
    enhance_mask_cloth_category: str = config.default_inpaint_mask_cloth_category
    
    enhance_mask_sam_model: str = config.default_inpaint_mask_sam_model
    
    enhance_mask_text_threshold: float = 0.25 # min 0.0 max 1.0
    enhance_mask_box_threshold: float = 0.30 # min 0.0 max 1.0
    enhance_mask_sam_max_detections: int = config.default_sam_max_detections # min 1 max 10, set 0 to detect all
    enhance_inpaint_disable_initial_latent: bool = False
    enhance_inpaint_engine: str = config.default_inpaint_engine_version
        
    enhance_inpaint_strength: float = 1.0 # min 0.0 max 1.0
    enhance_inpaint_respective_field: float = 0.618 # min 0.0 max 1.0
    enhance_inpaint_erode_or_dilate: int = 0 # min -64 max 64
    enhance_mask_inver: bool = False

    @field_validator("enhance_mask_model", mode="after")
    def validate_enhance_mask_model(cls, v):
        if v not in flags.inpaint_mask_models:
            raise ValueError(f"Invalid inpaint mask model: {v}")
        return v 
    
    @field_validator("enhance_inpaint_engine", mode="after")
    def validate_enhance_inpaint_engine(cls, v):
        if v not in flags.inpaint_engine_versions:
            raise ValueError(f"Invalid inpaint engine version: {v}")
        return v
    
    @field_validator("enhance_mask_sam_model", mode="after")
    def validate_enhance_mask_sam_model(cls, v):
        if v not in flags.inpaint_mask_sam_model:
            raise ValueError(f"Invalid inpaint mask sam model: {v}")
        return v
    
    @field_validator("enhance_mask_cloth_category", mode="after")
    def validate_enhance_mask_cloth_category(cls, v):
        if v not in flags.inpaint_mask_cloth_category:
            raise ValueError(f"Invalid inpaint mask cloth category: {v}")
        return v
    

class InpaintImputImage(BaseModel):
    image: Optional[numpy.ndarray] = Field(description="The image to inpaint")
    mask: Optional[numpy.ndarray] = Field(description="The mask to inpaint")

class _BaseImageGenerationObject(BaseModel):
    path_outputs: str = "./outputs"
    
    yields: list = []
    results: list = []
    patch_settings: dict[int, PatchSettings] = {}
    
    processing: bool = True
    last_stop: bool = False
    generate_image_grid: bool = False
    read_wildcards_in_order: bool = False # Read wildcards in order

class ImageGenerationSeed(_BaseImageGenerationObject):

    class Config:
        arbitrary_types_allowed = True
        orm_mode = True

    prompt: str = "A funny cat"
    negative_prompt: str = ""
    
    style_selections: list[str] = GLOBAL_CONFIG.default_styles
    performance_selection: str = GLOBAL_CONFIG.default_performance
    aspect_ratios_selection: str = GLOBAL_CONFIG.default_aspect_ratio
    base_model_name: str = Field("model.safetensors", description="Base model name")
    
    performance_loras: list = []
    original_steps: int = -1
    steps: int = -1
    
    image_number: int = Field(1, description="How many images to generate", ge=1)
    output_format: flags.LITERAL_OUTPUT_FORMATS = Field("png", description="Output format")
    
    seed: int = random.randint(config.MIN_SEED, config.MAX_SEED)
    sharpness: float = Field(2.0, description="Sharpness", ge=0.0, le=30.0)
    cfg_scale: float = Field(7.0, description="Higher value means style is cleaner, vivider, and more artistic.", ge=1.0, le=30.0)
    refiner_model_name: Optional[str] = Field(None, description="Refiner model name")
    refiner_switch: float = Field(0.8, description="Refiner switch", ge=0.0, le=1.0)
    
    loras: List[LoraTuple] = Field(config.GLOBAL_CONFIG.default_loras, description="LoRA settings")

    input_image_checkbox: bool = False
    current_tab: str = config.default_selected_image_input_tab_id
    
    uov_method: Optional[flags.UPSCALE_OR_VARIATION_MODES] = None
    uov_input_image: numpy.ndarray = None # TODO: trigger_auto_describe
    
    outpaint_selections: Optional[flags.OUTPAINT_SELECTIONS | list] = []
    inpaint_input_image: numpy.ndarray = None # TODO: trigger_auto_describe
    inpaint_additional_prompt: str = None
    inpaint_mask_image_upload: numpy.ndarray = None

    disable_preview: bool = False
    disable_intermediate_results: bool = False
    disable_seed_increment: bool = False

    # TODO type checks
    adm_scaler_positive: float = 1.5 # Min 0.1 max 3.0
    adm_scaler_negative: float = 0.8 # Min 0.1 max 3.0
    adm_scaler_end: float = 0.3 # Min 0.0 max 1.0
    
    adaptive_cfg: float = config.default_cfg_tsnr # min 1.0 max 30.0
    clip_skip: int = config.default_clip_skip # min 1 max config.clip_skip_max
    sampler_name: str = config.default_sampler
    scheduler_name: str = config.default_scheduler
    vae_name: str = config.default_vae
    overwrite_step: int = config.default_overwrite_step
    overwrite_switch: int = config.default_overwrite_switch
    overwrite_width: int = -1
    overwrite_height: int = -1
    overwrite_vary_strength: float = -1
    overwrite_upscale_strength: float = config.default_overwrite_upscale

    callback_steps: int = -1 # Used in async worker...

    should_upscale_or_vary: bool = False
    should_inpaint: bool = False
    should_use_imageprompt: bool = False
    
    mixing_image_prompt_and_vary_upscale: bool = False
    mixing_image_prompt_and_inpaint: bool = False
    debugging_cn_preprocessor: bool = False
    skipping_cn_preprocessor: bool = False
    canny_low_threshold: int = 64 # min 0 max 255

    canny_high_threshold: int = 128 # min 0 max 255
    refiner_swap_method: flags.REFINER_SWAP_METHODS = flags.refiner_swap_method
    controlnet_softness: float = 0.25 # min 0.0 max 1.0
    freeu_enabled: bool = False
    freeu_b1: float = 1.01 # min 0.0 max 2.0
    freeu_b2: float = 1.02 # min 0.0 max 2.0
    freeu_s1: float = 0.99 # min 0.0 max 2.0
    freeu_s2: float = 0.95 # min 0.0 max 2.0

    debugging_inpaint_preprocessor: bool = False
    inpaint_disable_initial_latent: bool = False
    inpaint_engine: str = config.default_inpaint_engine_version
    @field_validator("inpaint_engine", mode="after")
    def validate_inpaint_engine(cls, v):
        if v not in flags.inpaint_engine_versions:
            raise ValueError(f"Invalid inpaint engine version: {v}")
        return v
        
    


    inpaint_strength: float = 1.0 # min 0.0 max 1.0
    inpaint_respective_field: float = 0.618 # min 0.0 max 1.0
    inpaint_advanced_masking_checkbox: bool = config.default_inpaint_advanced_masking_checkbox
    invert_mask_checkbox: bool = False
    inpaint_erode_or_dilate: int = 0 # min -64 max 64
    
    save_final_enhanced_image_only: bool = bool(launch_arguments.args.disable_image_log)
    save_metadata_to_images: bool = config.default_save_metadata_to_images
    
    args_disable_metadata: bool = True

    cn_tasks: Optional[List[ControlNetImageTask]] | None = None

    debugging_dino: bool = False
    dino_erode_or_dilate: int = 0 # min -64 max 64
    debugging_enhance_masks_checkbox: bool = False

    enhance_input_image: numpy.ndarray = None
    enhance_checkbox: bool = Field(False, description="Enable enhacement")
    enhance_uov_method: Optional[flags.UPSCALE_OR_VARIATION_MODES] = Field(None, description="Enhacement uov method")
    enhance_uov_processing_order: flags.LITERAL_ENHANCEMENT_UOV_PROCESSING_ORDER = Field(flags.enhancement_uov_before, description="Enhacement uov processing order")

    enhance_uov_prompt_type: str = config.default_enhance_uov_prompt_type
    enhance_ctrls: Optional[list[EnhanceMaskCtrls]] = []
    should_enhance: bool = False
    def __init__(self, **data):
        super().__init__(**data)
        perf_name = flags.Performance(self.performance_selection).name
        self.steps = self.steps if self.steps != -1 else flags.Steps[perf_name].value
        self.original_steps = self.original_steps if self.original_steps != -1 else self.steps




