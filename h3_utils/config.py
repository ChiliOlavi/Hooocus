import os
import sys

from pydantic_core import from_json
from torch import Tensor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

import numpy

from h3_utils.flags import DESCRIBE_TYPE_PHOTO, ENHANCEMENT_UOV_PROMPT_TYPE_ORIGINAL, KSAMPLER, REFINER_SWAP_METHODS, SDXL_ASPECT_RATIOS, UPSCALE_OR_VARIATION_MODES, Overrides, Steps

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import json
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple, Iterable
from enum import Enum
from h3_utils.model_file_config import BaseControlNetTask
from pydantic import BaseModel, Field

from h3_utils.logging_util import LoggingUtil
from h3_utils.flags import EXAMPLE_ENHANCE_DETECTION_PROMPTS, INPAINT_MASK_CLOTH_CATEGORY, INPUT_IMAGE_MODES, KSAMPLER, OUTPAINT_SELECTIONS, REFINER_SWAP_METHODS, SDXL_ASPECT_RATIOS, UPSCALE_OR_VARIATION_MODES, LatentPreviewMethod, OutputFormat, Performance, ENHANCEMENT_UOV_AFTER, ENHANCEMENT_UOV_BEFORE, ENHANCEMENT_UOV_PROCESSING_ORDER

log = LoggingUtil().get_logger()

preset_chosen: str = "Realistic" # Modify this to change the preset
current_preset = {}

try:
    with open(f"{PARENT_DIR}/presets/default.json", "r") as f:
        DEFAULT_PRESET = json.load(f)
        log.info("Default preset loaded.")
except FileNotFoundError:
    raise FileNotFoundError("Could not find default preset file. Exiting.")


try:
    with open(f"{PARENT_DIR}/presets/{preset_chosen.lower()}.json", "r") as f:
        log.info(f"Loading preset file for {preset_chosen}.")
        current_preset = json.load(f)
except FileNotFoundError:
    log.error(f"Could not find preset file for {preset_chosen}. Using default preset.")
    current_preset = DEFAULT_PRESET

class GlobalEnv:
    # NB! Do not store any sensitive information here. Use normal .env files for that.

    PYTHONFAULTHANDLER=1

    # launch.py
    REINSTALL_ALL = False
    TRY_INSTALL_XFORMERS = False

    PYTORCH_ENABLE_MPS_FALLBACK = 1
    PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0

    def __init__(self, **data):
        # Set the environment variables globally
        super().__init__(**data)
        for _key, value in data.items():
            os.environ[_key] = str(value)


HOOOCUS_VERSION = '0.5.1'
METADATA_SCHEME = "Hooocus"



class _LAUNCH_ARGS(BaseModel):
    # Modify the initial values here
    class Config:
        arbitrary_types_allowed = True

   
    # General args
    enable_auto_describe_image: bool = Field(False, description="Enables automatic description of uov and enhance image when prompt is empty.")
    preview_option: LatentPreviewMethod = LatentPreviewMethod.NoPreviews
    wildcards_max_bfs_depth: int = 64
    disable_image_log: bool = Field(False, description="Prevent writing images and logs to the outputs folder.")
    disable_analytics: bool = Field(False, description="Disables analytics for Gradio.")
    disable_metadata: bool = Field(False, description="Disables saving metadata to images.")
    disable_preset_download: bool = Field(False, description="Disables downloading models for presets.")
    disable_enhance_output_sorting: bool = Field(False, description="Disables enhance output sorting for final image gallery.")
    always_download_new_model: bool = Field(False, description="Always download newer models.")
    rebuild_hash_cache: bool = Field(False, description="Generates missing model and LoRA hashes.")
    temp_path_cleanup_on_launch: bool = Field(True, description="The temp path cleanup on launch to use.")
    
    # Server args
    listen: str = "127.0.0.1"
    port: int = 8188
    disable_header_check: str = "false"
    disable_server_info: bool = False
    
    # Etc
    web_upload_size: float = 100.0
    hf_mirror: str = "https:/huggingface.co"
    external_working_path: str = None
    temp_path: str = None
    cache_path: str = None
    in_browser: bool = False
    disable_in_browser: bool = False

    # Global imagegen
    min_seed: int = 0
    max_seed: int = 2**63 - 1
    black_out_nsfw: bool = False
    disable_attention_upcast: bool = False
    gpu_device_id: Optional[int] = None
    output_path: str = None
    directml: bool = False
    disable_ipex_hijack: bool = False
    disable_xformers: bool = False
    pytorch_deterministic: bool = False
    

    # CMD args
    async_cuda_allocation: bool = False
    disable_async_cuda_allocation: bool = False

    # Model args
    all_in_fp32: bool = False
    all_in_fp16: bool = False

    # Unet args
    unet_in_bf16: bool = False
    unet_in_fp16: bool = False
    unet_in_fp8_e4m3fn: bool = False
    unet_in_fp8_e5m2: bool = False

    # VAE args
    vae_in_fp16: bool = False
    vae_in_fp32: bool = False
    vae_in_bf16: bool = False
    vae_in_cpu: bool = False

    # FPTEArgs
    clip_in_fp8_e4m3fn: bool = False
    clip_in_fp8_e5m2: bool = False
    clip_in_fp16: bool = False
    clip_in_fp32: bool = False

    # AttentionArgs
    attention_split: bool = False
    attention_quad: bool = False
    attention_pytorch: bool = False

    # VramArgs
    always_cpu: int = False
    always_gpu: int = True
    always_high_vram: int = -1
    always_normal_vram: int = 1
    always_low_vram: int = -1
    always_no_vram: int = -1
    always_offload_from_vram: bool = False



LAUNCH_ARGS = _LAUNCH_ARGS()

class FilePathConfig:
    config_path = 'h3_utils/config.json'
    hash_cache_path = f'{PARENT_DIR}/__cache__/hash_cache.json'
    auth_filename = 'auth.json'








class FreeUControls(BaseModel):
    freeu_b1: float = Field(1.01, le=2.0, ge=0.0)
    freeu_b2: float = Field(1.02, le=2.0, ge=0.0)
    freeu_s1: float = Field(0.99, le=2.0, ge=0.0)
    freeu_s2: float = Field(0.95, le=2.0, ge=0.0)

class OverWriteControls(BaseModel):
    overwrite_height: int = -1
    overwrite_step: int = -1
    overwrite_switch: int = -1
    overwrite_upscale_strength: float = -1
    overwrite_upscale: float = -1
    overwrite_vary_strength: float = -1
    overwrite_width: int = -1

class DeveloperOptions(BaseModel):
     # ?
    metadata_created_by: str = Field("", description="The metadata created by to use.")
    metadata_scheme: str = Field(METADATA_SCHEME, description="The default metadata scheme to use.")
    debugging_cn_preprocessor: bool = False
    debugging_dino: bool = False
    debugging_enhance_masks_checkbox: bool = False
    debugging_inpaint_preprocessor: bool = False
    temp_path_cleanup_on_launch: bool = Field(True, description="The temp path cleanup on launch to use.")
    temp_path: str = Field(os.path.join(tempfile.gettempdir(), 'hooocus'), description="The default temp path to use.")
    disable_intermediate_results: bool = False
    disable_seed_increment: bool = False
    generate_grid: bool = False
    disable_preview: bool = False
    skipping_cn_preprocessor: bool = False
    should_use_advanced: bool = Field(False, description="Bool: should use advanced checkbox?")
    should_use_developer_debug_mode: bool = Field(False, description="Bool: default developer debug mode.")

class InptaintOptions(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    inpaint_worker_current_task: Optional[Any] = None
    outpaint_selections: Optional[OUTPAINT_SELECTIONS] = None
    invert_mask: bool = False
    inpaint_mask_model: str = Field('isnet-general-use', description="The default invert mask model to use.")
    inpaint_additional_prompt: Optional[str] = None
    inpaint_disable_initial_latent: bool = False
    inpaint_engine_version: str = Field("v2.6", description="The default inpaint engine version to use.")
    inpaint_erode_or_dilate: Optional[int] = Field(None, le=64, ge=-64)
    inpaint_mask_cloth_category: str = Field('full', description="The default inpaint mask cloth category to use.")
    inpaint_mask_sam_model: str = Field('vit_b', description="The default inpaint mask sam model to use.")
    inpaint_method: str = Field("Inpaint or Outpaint (default)", description="The default inpaint method to use.")
    inpaint_respective_field: float = 0.618 # min 0.0 max 1.0
    use_advanced_inpaint_masking: bool = Field(False, description="Bool: should use advanced inpaint masking?")
    inpaint_should_use_inpaint_image: bool = False
    inpaint_should_use_mask: bool = False
    inpaint_stop_ats: List[float] = Field([0.5, 0.5, 0.5, 0.5], description="The default inpaint stop ats to use.")
    inpaint_strength: float = 1.0 # min 0.0 max 1.0
    

class EnhanceMaskCtrls(BaseModel):
    """
    Enhacement mask controls for inpaint and outpaint
    
    """
    class Config:
        arbitrary_types_allowed = True

    enhance_tabs: int = Field(3, description="The default enhance tabs to use.")
    enhance_mask_dino_prompt_text: str = EXAMPLE_ENHANCE_DETECTION_PROMPTS[0]
    
    enhance_prompt: str = None # Enhacement positive prompt. Uses original prompt if None
    enhance_negative_prompt: str = None # Enhacement negative prompt. Uses original negative prompt if None
    
    enhance_inpaint_mask_model: str = Field('sam', description="The default enhance inpaint mask model to use.")
    enhance_mask_cloth_category: INPAINT_MASK_CLOTH_CATEGORY = 'full'
    
    enhance_mask_sam_model: str = Field('vit_b', description="The default inpaint mask sam model to use.")
    
    enhance_mask_text_threshold: float = Field(0.25, description="The default enhance mask text threshold to use.", ge=0.0, le=1.0)
    enhance_mask_box_threshold: float = Field(0.30, description="The default enhance mask box threshold to use.", ge=0.0, le=1.0)
    enhance_mask_sam_max_detections: int = Field(0, description="The default enhance mask sam max detections to use.", ge=0, le=10)
    
    enhance_inpaint_disable_initial_latent: bool = False
    enhance_inpaint_engine: str = Field("2.6", description="The default enhance inpaint engine version to use.")
        
    enhance_inpaint_strength: float = Field(1.0, description="The default enhance inpaint strength to use.", ge=0.0, le=1.0)
    enhance_inpaint_respective_field: float = Field(0.618, description="The default enhance inpaint respective field to use.", ge=0.0, le=1.0)
    enhance_inpaint_erode_or_dilate: int = Field(0, description="The default enhance inpaint erode or dilate to use.", ge=-64, le=64)
    enhance_mask_inver: bool = False

    enhance_uov_method: Optional[UPSCALE_OR_VARIATION_MODES] = Field(None, description="Enhacement uov method")
    enhance_uov_method: Optional[str] = Field(None, description="The default enhance uov method to use.")
    enhance_uov_processing_order: int = Field(ENHANCEMENT_UOV_BEFORE, description="The default enhance uov processing order to use.")
    enhance_uov_prompt_type: int = Field(ENHANCEMENT_UOV_PROMPT_TYPE_ORIGINAL, description="The default enhance uov prompt type to use.")
    

class _InitialImageGenerationParams(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    prompt_negative: str = Field(DEFAULT_PRESET["prompt_negative"], description="The default negative prompt to use.")
    prompt: Optional[str] = Field(None, description="The default prompt to use.")
    negative_prompt: str = ""
    read_wildcards_in_order: bool = False

    width: Optional[int] = Field(None, description="The default width to use.")
    height: Optional[int] = Field(None, description="The default height to use.")
    
    sample_sharpness: float = Field(DEFAULT_PRESET["sample_sharpness"], description="The default sample sharpness to use.")
    seed: int = random.randint(LAUNCH_ARGS.min_seed, LAUNCH_ARGS.max_seed)
    #sharpness: float = Field(2.0, description="Sharpness", ge=0.0, le=30.0)
    sampler_name: KSAMPLER = DEFAULT_PRESET["sampler"]
    scheduler_name: str = DEFAULT_PRESET["scheduler"]
    
    base_model_name: str = Field(DEFAULT_PRESET["model"], description="The default model to use.", alias="model")
    refiner_model: str = Field(DEFAULT_PRESET["refiner"], description="The default refiner model to use.", )
    refiner_switch: float = Field(DEFAULT_PRESET["refiner_switch"], description="Refiner switch", ge=0.0, le=1.0)
    refiner_swap_method: REFINER_SWAP_METHODS = "joint"
    loras: list = Field(DEFAULT_PRESET["loras"], description="The default LoRAs to use.")
    styles: List[str] = Field(DEFAULT_PRESET["styles"], description="The default styles to use.")
    
    vae_name: str = Field("Default (model)", description="The default vae to use.")
    
    performance_selection: Performance = Performance.SPEED
    performance_loras: list = []

    previous_default_models: List[str] = Field(DEFAULT_PRESET["previous_default_models"], description="The default previous default models to use.")
    
    # Format and save options
    output_format: OutputFormat = Field("png", description="Output format")
    save_metadata_to_images: bool = Field(False, description="The default save metadata to images to use.")
    save_only_final_enhanced_image: bool = Field(False, description="The default save only final enhanced image to use.")
    aspect_ratio: SDXL_ASPECT_RATIOS = Field(DEFAULT_PRESET["aspect_ratio"], description="The default aspect ratio to use.")
    image_number: int = Field(2, description="The default image number to use.", ge=1)
    
    steps: int = -1
    original_steps: int = -1

    adaptive_cfg: float = Field(7.0, description="The default cfg tsnr to use.", ge=1.0, le=30.0)
    cfg_scale: float = Field(DEFAULT_PRESET["cfg_scale"], description="Higher value means style is cleaner, vivider, and more artistic.", ge=1.0, le=30.0)
    cfg_tsnr: float = Field(7.0, description="The default cfg tsnr to use.")
    
    adm_scaler_end: float = Field(0.3, description="The default adm scaler end to use.", ge=0.0, le=1.0)
    adm_scaler_negative: float = Field(0.8, description="The default adm scaler negative to use.", ge=0.1, le=3.0)
    adm_scaler_positive: float = Field(1.5, description="The default adm scaler positive to use.", ge=0.1, le=3.0)
    
    
    canny_high_threshold: int = Field(128, description="The default canny high threshold to use.", ge=0, le=255)
    canny_low_threshold: int = Field(64, description="The default canny low threshold to use.", ge=0, le=255)
    
    checkpoint_downloads: dict[str, str] = Field(DEFAULT_PRESET["checkpoint_downloads"], description="The default checkpoint downloads to use.")
    vae_downloads: dict[str, str] = Field({}, description="The default vae downloads to use.")
    lora_downloads: dict[str, str] = Field({}, description="The default lora downloads to use.")
    embeddings_downloads: dict[str, str] = Field(DEFAULT_PRESET["embeddings_downloads"], description="The default embeddings downloads to use.")

    clip_skip: int = Field(2, description="The default clip skip to use.") # Where clip skip max?
    controlnet_softness: float = Field(0.25, description="The default controlnet softness to use.", ge=0.0, le=1.0)
    dino_erode_or_dilate: int = 0 # min -64 max 64

    
    enhance_task: Optional[EnhanceMaskCtrls] = None
    freeu_controls: Optional[FreeUControls] = None
    inpaint_options: Optional[InptaintOptions] = None
    controlnet_tasks: Optional[List[BaseControlNetTask]] = None
    overwrite_controls: Optional[OverWriteControls] = None
    developer_options: Optional[DeveloperOptions] = DeveloperOptions()
    
    #should_describe_apply_prompts: bool = Field(True, description="The default describe apply prompts checkbox to use.")
    describe_content_type: Optional[List[str]] = Field([DESCRIBE_TYPE_PHOTO], description="The default describe content type to use.")
    
    use_image_input: bool = Field(False, description="Bool: should use image input?")
    use_image_prompt_advanced: bool = Field(False, description="Bool: should use image prompt advanced?")
    use_imageprompt: bool = False
    use_upscale_or_vary: bool = False
    mix_image_prompt_and_vary_upscale: bool = False
    mix_image_prompt_and_inpaint: bool = False

    image_input_mode: INPUT_IMAGE_MODES = Field("uov", description="The image input mode to use.") # utils.flags.input_image_tab_ids 
    
    input_image: Optional[Dict[Literal["image", "mask"], numpy.ndarray]] = None
    uov_input_image: Optional[numpy.ndarray] = None
    input_mask_image: Optional[Dict[Literal["image", "mask"], numpy.ndarray]] = None
    prepared_input_mask_image: Optional[numpy.ndarray] = None
    enhance_input_image: Optional[numpy.ndarray] = None
    
    uov_method: Optional[UPSCALE_OR_VARIATION_MODES] = Field(None, description="The default uov method to use.")
    steps_uov: int = -1
    


class TaskletObject(BaseModel):
    """A tasklet object for the async worker.
    
    - It's a variation of a ImageGenerationObject 
    with additional fields for the pipeline.process_diffusion method.

    - So if you have an ImageGenerationObject with image_count=2,
    you will have 2 TaskletObjects with the same data but some
    variations for the pipeline.process_diffusion method.
    """
    class Config:
        arbitrary_types_allowed = True
        orm_mode = True


    task_seed: int 
    task_prompt: str
    task_negative_prompt: str
    positive_basic_workloads: Optional[List[str]] = []
    negative_basic_workloads: Optional[List[str]] = []
    expansion: str
    # [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]
    encoded_positive_cond: Optional[Tuple[Tensor, Dict[str, Tensor]]] = None
    encoded_negative_cond: Optional[Tuple[Tensor, Dict[str, Tensor]]] = None
    positive_top_k: int = 0
    negative_top_k: int = 0
    log_positive_prompt: str 
    log_negative_prompt: str
    styles: List[str]

class ApplyImageInputParams(BaseModel):
    base_model_additional_loras: List[str]
    clip_vision_path: str
    controlnet_canny_path: str
    controlnet_cpds_path: str
    inpaint_head_model_path: str
    inpaint_image: str
    inpaint_mask: str
    ip_adapter_face_path: str
    ip_adapter_path: str
    ip_negative_path: str
    skip_prompt_processing: bool
    use_synthetic_refiner: bool

class ImageGenerationObject(_InitialImageGenerationParams):
    
    class Config:
        arbitrary_types_allowed = True
        orm_mode = True
    
    
HooocusConfig = ImageGenerationObject(**current_preset)
DefaultConfigImageGen = ImageGenerationObject(**DEFAULT_PRESET)

print(HooocusConfig.dict())