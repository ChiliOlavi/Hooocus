from modules.async_worker import ImageTaskProcessor
import ldm_patched.modules.model_management
from h3_utils.config import ImageGenerationObject
from unavoided_globals.global_model_management import global_model_management

import time

def generate_image(task: ImageGenerationObject) -> ImageTaskProcessor:
    

    try:
        with global_model_management.interrupt_processing_mutex:
            global_model_management.interrupt_processing = False
        
        print(task.refiner_model)
        imgProcessor = ImageTaskProcessor()
        imgProcessor.generation_tasks.append(task)
        finished = False

        time.sleep(0.01)
        imgProcessor.process_all_tasks()
        print('Image generation finished.')
        ...

    except Exception as e:
        print(str(e))

    finally:
        print('Done')
        return imgProcessor