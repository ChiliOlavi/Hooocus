from h3_utils.launch import prepare_environment
prepare_environment()

from modules.async_worker import ImageTaskProcessor
import ldm_patched.modules.model_management as model_management
from h3_utils.config import ImageGenerationObject, EnhanceMaskCtrls

import time

def generate_image(task: ImageGenerationObject):

    try:
        with model_management.interrupt_processing_mutex:
            model_management.interrupt_processing = False
        
        print(task.refiner_model)
        imgProcessor = ImageTaskProcessor()
        imgProcessor.generation_tasks.append(task)
        finished = False

        time.sleep(0.01)
        imgProcessor.process_all_tasks()
        """ if len(imgProcessor.yields) > 0:
            flag, product = imgProcessor.yields.pop(0)
            if flag == 'preview':
                # help bad internet connection by skipping duplicated preview
                if len(imgProcessor.yields) > 0:  # if we have the next item
                    if imgProcessor.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                print(f'{percentage:.2f}%: {title}')
                ...
            if flag == "results":
                ...
            if flag == 'finish':
                finished = True """
        print('Image generation finished.')
        ...

    except Exception as e:
        print(str(e))

    finally:
        print('Done')

if __name__ == '__main__':

    image_params = ImageGenerationObject(
        prompt="A car without headlights a banana"
        )
    generate_image(image_params)
    print('Done fom main.py')