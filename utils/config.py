import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import json
import tempfile
from typing import List, Optional, Union
from enum import Enum

from pydantic import BaseModel, field_validator
from pydantic import BaseModel, Field

from utils.hooocus_utils import LoraTuple
from utils.logging_util import LoggingUtil
from utils.flags import Performance

log = LoggingUtil().get_logger()

class FilePathConfig(Enum):
    config_path = 'utils/config.json'
    hash_cache_path = f'{PARENT_DIR}/__cache__/hash_cache.json'
    auth_filename = 'auth.json'

class PathsConfig(Enum):
    path_checkpoints = "./models/checkpoints/"
    path_loras = "./models/loras/"
    path_embeddings = "./models/embeddings/"
    path_vae_approx = "./models/vae_approx/"
    path_vae = "./models/vae/"
    path_upscale_models = "./models/upscale_models/"
    path_inpaint = "./models/inpaint/"
    path_controlnet = "./models/controlnet/"
    path_clip_vision = "./models/clip_vision/"
    path_fooocus_expansion = "./models/prompt_expansion/fooocus_expansion"
    path_wildcards = "./wildcards/"
    path_safety_checker = "./models/safety_checker/"
    path_sam = "./models/sam/"
    default_temp_path = os.path.join(tempfile.gettempdir(), 'hooocus')

    def __init__(self, **data):
        super().__init__(**data)
        for _key, value in data.items():
            if ".json" in value:
                pass
            if not os.path.exists(value):
                log.error(f"Path does not exist for: {value}. Creating it.")
                os.makedirs(value)


class GlobalEnv(Enum):
    PYTHONFAULTHANDLER=1
    HOOOCUS_VERSION = '0.5.0'

    # launch.py
    REINSTALL_ALL = False
    TRY_INSTALL_XFORMERS = False

    # From launch.py
    PYTORCH_ENABLE_MPS_FALLBACK = 1
    PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        for _key, value in data.items():
            os.environ[_key] = str(value)


class LaunchArguments(BaseModel):
    disable_offload_from_vram: bool = Field(False, description="Force loading models to vram when the unload can be avoided.")
    disable_image_log: bool = Field(False, description="Prevent writing images and logs to the outputs folder.")
    disable_analytics: bool = Field(False, description="Disables analytics for Gradio.")
    disable_metadata: bool = Field(False, description="Disables saving metadata to images.")
    disable_preset_download: bool = Field(False, description="Disables downloading models for presets.")
    disable_enhance_output_sorting: bool = Field(False, description="Disables enhance output sorting for final image gallery.")
    enable_auto_describe_image: bool = Field(False, description="Enables automatic description of uov and enhance image when prompt is empty.")
    always_download_new_model: bool = Field(False, description="Always download newer models.")
    rebuild_hash_cache: bool = Field(False, description="Generates missing model and LoRA hashes.")

    min_seed: int = 0
    max_seed: int = 2**63 - 1

    metadata_scheme: str = "Hooocus"
    preset_chosen: str = "Realistic"
    current_preset = {}
    performance_selection = Performance.SPEED

    def __init__(self, **data):
        try:
            with open(f"{PARENT_DIR}/presets/{self.preset_chosen.lower()}.json") as f:
                self.current_preset = json.load(f)
        except FileNotFoundError:
            log.error(f"Could not find preset file for {self.preset_chosen}.")
            self.current_preset = {}
        
        super().__init__(**data) # Call the parent class constructor

LAUNCH_ARGUMENTS = LaunchArguments()

class HocusConfig(BaseModel):
    default_model: str = Field(LAUNCH_ARGUMENTS.current_preset["default_model"], description="The default model to use.")
    default_refiner: str = Field(LAUNCH_ARGUMENTS.current_preset["default_refiner"], description="The default refiner model to use.")
    default_refiner_switch: int = Field(LAUNCH_ARGUMENTS.current_preset["default_refiner_switch"], description="The default refiner switch to use.")
    default_loras: List[LoraTuple] = Field(LAUNCH_ARGUMENTS.current_preset["default_loras"], description="The default LoRAs to use.")
    
    @field_validator("default_loras", mode="before")
    def validate_default_loras(cls, v):
        loras_to_add = [LoraTuple(**lora) for lora in v]
        return loras_to_add

    default_cfg_scale: float = Field(LAUNCH_ARGUMENTS.current_preset["default_cfg_scale"], description="The default cfg scale to use.")
    default_sample_sharpness: float = Field(LAUNCH_ARGUMENTS.current_preset["default_sample_sharpness"], description="The default sample sharpness to use.")
    default_sampler: str = Field(LAUNCH_ARGUMENTS.current_preset["default_sampler"], description="The default sampler to use.")
    default_scheduler: str = Field(LAUNCH_ARGUMENTS.current_preset["default_scheduler"], description="The default scheduler to use.")
    default_performance: str = Field(LAUNCH_ARGUMENTS.current_preset["default_performance"], description="The default performance to use.")
    default_prompt: str = Field(LAUNCH_ARGUMENTS.current_preset["default_prompt"], description="The default prompt to use.")
    default_prompt_negative: str = Field(LAUNCH_ARGUMENTS.current_preset["default_prompt_negative"], description="The default negative prompt to use.")
    default_styles: List[str] = Field(LAUNCH_ARGUMENTS.current_preset["default_styles"], description="The default styles to use.")
    default_aspect_ratio: str = Field(LAUNCH_ARGUMENTS.current_preset["default_aspect_ratio"], description="The default aspect ratio to use.")
    default_overwrite_step: int = Field(LAUNCH_ARGUMENTS.current_preset["default_overwrite_step"], description="The default overwrite step to use.")
    checkpoint_downloads: dict[str, str] = Field(LAUNCH_ARGUMENTS.current_preset["checkpoint_downloads"], description="The default checkpoint downloads to use.")
    embeddings_downloads: dict[str, str] = Field(LAUNCH_ARGUMENTS.current_preset["embeddings_downloads"], description="The default embeddings downloads to use.")
    previous_default_models: List[str] = Field(LAUNCH_ARGUMENTS.current_preset["previous_default_models"], description="The default previous default models to use.")
    
    default_loras_min_weight: float = Field(2, description="The default loras min weight to use.")
    default_loras_max_weight: float = Field(2, description="The default loras max weight to use.")
    default_max_lora_number: int = Field(5, description="The default max lora number to use.")
    default_vae: str = Field("Default (model)", description="The default vae to use.")
    
    should_use_image_input: bool = Field(False, description="Bool: should use image input?")
    should_use_image_prompt_advanced: bool = Field(False, description="Bool: should use image prompt advanced?")
    should_enhance_image: bool = Field(False, description="Bool: should enhance image?")
    should_use_advanced: bool = Field(False, description="Bool: should use advanced checkbox?")
    should_use_developer_debug_mode: bool = Field(False, description="Bool: default developer debug mode.")
    should_use_advanced_inpaint_masking: bool = Field(False, description="Bool: should use advanced inpaint masking?")
    
    default_controlnet_image_count: int = Field(4, description="The default controlnet image count to use.")
    default_max_image_number: int = Field(32, description="The default max image number to use.")
    default_image_number: int = Field(2, description="The default image number to use.")
    lora_downloads: dict[str, str] = Field({}, description="The default lora downloads to use.")
    vae_downloads: dict[str, str] = Field({}, description="The default vae downloads to use.")
    available_aspect_ratios: List[str] = Field(utils.flags.sdxl_aspect_ratios, description="The available aspect ratios to use.")
    default_inpaint_engine_version: str = Field("v2.6", description="The default inpaint engine version to use.")
    image_input_mode: str = Field("uov_tab", description="The image input mode to use.") # utils.flags.input_image_tab_ids 
    default_inpaint_method: str = Field("Inpaint or Outpaint (default)", description="The default inpaint method to use.")
    default_uov_method: Optional[str] = Field(None, description="The default uov method to use.")
    
    default_cfg_tsnr: float = Field(7.0, description="The default cfg tsnr to use.")
    default_clip_skip: int = Field(2, description="The default clip skip to use.")
    default_overwrite_switch: int = Field(-1, description="The default overwrite switch to use.")
    default_overwrite_upscale: float = Field(-1, description="The default overwrite upscale to use.")



    default_enhance_tabs: int = Field(3, description="The default enhance tabs to use.")
    default_enhance_uov_method: Optional[str] = Field(None, description="The default enhance uov method to use.")
    default_enhance_uov_processing_order: int = Field(utils.flags.enhancement_uov_before, description="The default enhance uov processing order to use.")
    default_enhance_uov_prompt_type: int = Field(utils.flags.enhancement_uov_prompt_type_original, description="The default enhance uov prompt type to use.")
    default_sam_max_detections: int = Field(0, description="The default sam max detections to use.")
    default_black_out_nsfw: bool = Field(False, description="The default black out nsfw to use.")
    
    default_save_only_final_enhanced_image: bool = Field(False, description="The default save only final enhanced image to use.")
    default_save_metadata_to_images: bool = Field(False, description="The default save metadata to images to use.")
    default_metadata_scheme: str = Field("Hooocus", description="The default metadata scheme to use.")
    metadata_created_by: str = Field("", description="The metadata created by to use.")

    should_use_invert_mask: bool = Field(False, description="Bool: should use invert mask checkbox?")
    default_invert_mask_model: str = Field('isnet-general-use', description="The default invert mask model to use.")
    default_enhance_inpaint_mask_model: str = Field('sam', description="The default enhance inpaint mask model to use.")
    default_inpaint_mask_cloth_category: str = Field('full', description="The default inpaint mask cloth category to use.")
    default_inpaint_mask_sam_model: str = Field('vit_b', description="The default inpaint mask sam model to use.")
    default_inpaint_stop_ats: List[float] = Field([0.5, 0.5, 0.5, 0.5], description="The default inpaint stop ats to use.")
    default_describe_apply_prompts_checkbox: bool = Field(True, description="The default describe apply prompts checkbox to use.")
    default_describe_content_type: List[str] = Field([utils.flags.describe_type_photo], description="The default describe content type to use.")

    default_temp_path: str = Field(os.path.join(tempfile.gettempdir(), 'hooocus'), description="The default temp path to use.")
    temp_path_cleanup_on_launch: bool = Field(True, description="The temp path cleanup on launch to use.")

GLOBAL_CONFIG = HocusConfig(**LAUNCH_ARGUMENTS.current_preset)


