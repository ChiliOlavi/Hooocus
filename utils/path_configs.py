
import os
import tempfile
from enum import Enum

class FolderPathsConfig(Enum):
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
    path_outputs: str = "./outputs"


for path in FolderPathsConfig:
    if not os.path.exists(path.value):
        os.makedirs(path.value)
        log.info(f"Created path: {path.value}")