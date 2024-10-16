import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USE_GRADIO = 0
PYTHONFAULTHANDLER=1
HOOOCUS_VERSION = '0.5.0'

CHECK_UPDATES=False # Check git upstream for updates
HASH_CACHE_PATH = f'{PARENT_DIR}/__cache__/hash_cache.json'

# launch.py
REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False

# From launch.py
PYTORCH_ENABLE_MPS_FALLBACK = 1
PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0
GRADIO_SERVER_PORT = 7865

# From old modules/constants.py
# as in k-diffusion (sampling.py)
MIN_SEED = 0
MAX_SEED = 2**63 - 1

AUTH_FILENAME = 'auth.json'

# From old modules/constants.py
PYTORCH_ENABLE_MPS_FALLBACK = 1

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
