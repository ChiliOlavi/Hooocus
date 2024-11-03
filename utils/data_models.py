import numpy
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Union


CONTROLNET_TASK_TYPES = Literal["ImagePrompt", "FaceSwap", "PyraCanny", "CPDS"]    

UPSCALE_OR_VARIATION_MODELS = Literal[
        'Enabled',
        'Vary (Subtle)',
        'Vary (Strong)',
        'Upscale (1.5x)',
        'Upscale (2x)',
        'Upscale (Fast 2x)',
]
