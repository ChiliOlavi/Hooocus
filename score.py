import logging
import io
import json
from base64 import b64encode
from azureml.contrib.services.aml_response import AMLResponse
from h3_utils.launch import prepare_environment
from h3_utils.config import ImageGenerationObject
from .modules.async_worker import ImageTaskProcessor

#Thanks to Azure/gen-cv for the implementation: https://github.com/Azure/gen-cv/tree/main?tab=MIT-1-ov-file#readme
def init():
    prepare_environment()



def encode_response(images):
    """
    This function takes a list of images and converts them to a dictionary of base64 encoded strings.
    """
    ENCODING = 'utf-8'
    dic_response = {}
    for i, image in enumerate(images):
        output = io.BytesIO()
        image.save(output, format="JPEG")
        base64_bytes = b64encode(output.getvalue())
        base64_string = base64_bytes.decode(ENCODING)
        dic_response[f'image_{i}'] = base64_string

    return dic_response

def run(raw_data):
   data = json.loads(raw_data)["data"]
   img_obj = ImageGenerationObject.model_validate(data)
   img_generator = ImageTaskProcessor.process_single_task(img_obj)
   if len(img_generator.yields)>0:
      flag, product = img_generator.yields.pop(0)
      if flag=="finish":
        encoded_response = encode_response(product)
   resp = AMLResponse(message=encoded_response, status_code=200, json_str=True)
   return resp





    	