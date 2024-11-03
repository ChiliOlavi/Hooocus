import numpy
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Union


class ControlNetTask(BaseModel):
    cn_img: numpy.ndarray | None = Field(None, description="The ndarray format of the image for the ControlNetTask.")
    cn_stop: float = Field(None, description="Stop point of the ControlNetTask.", ge=0, le=1)
    cn_weight: float = Field(None, description="Weight of the ControlNetTask.", ge=0, le=1)
    cn_type: CONTROLNET_TASK_TYPES = Field("ImagePrompt", description="Type of the ControlNetTask.")

    class Config:
        arbitrary_types_allowed = True


class Default:
    ImagePrompt = ControlNetTask(cn_type="ImagePrompt", cn_stop=0.5, cn_weight=0.6)
    ImagePromptFaceSwap = ControlNetTask(cn_type="FaceSwap", cn_stop=0.9, cn_weight=0.75)
    PyraCanny = ControlNetTask(cn_type="PyraCanny", cn_stop=0.5, cn_weight=1.0)
    CPDS = ControlNetTask(cn_type="CPDS", cn_stop=0.5, cn_weight=1.0)
