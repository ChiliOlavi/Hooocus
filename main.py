from math import e
from utils.launch import *
import modules.async_worker as worker
import ldm_patched.modules.model_management as model_management
from modules.hookus_utils import ImageGenerationSeed, EnhanceMaskCtrls, ControlNetImageTask, LoraTuple

def generate_image(task: ImageGenerationSeed):

    try:
        with model_management.interrupt_processing_mutex:
            model_management.interrupt_processing = False
        
        workhorse = worker
        workhorse.async_tasks.append(task)
        finished = False

        while not finished:
            time.sleep(0.01)
            if len(task.yields) > 0:
                flag, product = task.yields.pop(0)
                if flag == 'preview':

                    # help bad internet connection by skipping duplicated preview
                    if len(task.yields) > 0:  # if we have the next item
                        if task.yields[0][0] == 'preview':   # if the next item is also a preview
                            # print('Skipped one preview for better internet connection.')
                            continue

                    percentage, title, image = product
                    print(f'{percentage:.2f}%: {title}')
                    ...
                if flag == "results":
                    ...
                if flag == 'finish':
                    finished = True
        print('Image generation finished.')
        ...

    except Exception as e:
        print(str(e))

    finally:
        print('Done')