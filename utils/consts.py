import json
import os, sys

from pydantic import BaseModel, Field
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from utils.logging_util import LoggingUtil

log = LoggingUtil().get_logger()


USE_GRADIO = 0
PYTHONFAULTHANDLER=1
HOOOCUS_VERSION = '0.5.0'
CONFIG_PATH = 'utils/config.json'





class PathsConfig(BaseModel):
    path_checkpoints: list = Field(default='../models/checkpoints/')
    path_loras: list = Field(default='../models/loras/')
    path_embeddings: str = Field(default='../models/embeddings/')
    path_vae_approx: str = Field(default='../models/vae_approx/')
    path_vae: str = Field(default='../models/vae/')
    path_upscale_models: str = Field(default='../models/upscale_models/')
    path_inpaint: str = Field(default='../models/inpaint/')
    path_controlnet: str = Field(default='../models/controlnet/')
    path_clip_vision: str = Field(default='../models/clip_vision/')
    path_fooocus_expansion: str = Field(default='../models/prompt_expansion/fooocus_expansion')
    path_wildcards: str = Field(default='../wildcards/')
    path_safety_checker: str = Field(default='../models/safety_checker/')
    path_sam: str = Field(default='../models/sam/')

    def __init__(self, **data):
        super().__init__(**data)
        for _key, value in data.items():
            if not os.path.exists(value):
                log.error(f"Path does not exist for: {value}. Creating it.")
                os.makedirs(value)

DEFAULT_PATHS_CONFIG = PathsConfig()

CHECK_UPDATES=False # Check git upstream for updates
HASH_CACHE_PATH = f'{PARENT_DIR}/__cache__/hash_cache.json'

# launch.py
REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False

# From launch.py
PYTORCH_ENABLE_MPS_FALLBACK = 1
PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0


AUTH_FILENAME = 'auth.json'

# Set globals for os.environ 
locals = vars().copy()
LOCALS = []
for k, v in locals.items():
    if k.isupper():
        LOCALS.append({k: v})


for local in LOCALS:
    for k, v in local.items():
        os.environ[k] = str(v)

...
