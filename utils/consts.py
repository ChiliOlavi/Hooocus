import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


HOOOCUS_VERSION = '2.5.5'

CHECK_UPDATES=False # Check git upstream for updates
HASH_CACHE_PATH = f'{PARENT_DIR}/__cache__/hash_cache.json'

# launch.py
REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False