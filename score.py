import logging
import io
import json
from base64 import b64encode
from azureml.contrib.services.aml_response import AMLResponse
from main import generate_image()
from h3_utils.launch import prepare_environment
from h3_utils.config import ImageGenerationObject

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
   new_imgs = generate_image(img_obj) #Does not really return the image at this point.
   encoded_response = encode_response(new_imgs)
   resp = AMLResponse(message=encoded_response, status_code=200, json_str=True)
   return resp





    	