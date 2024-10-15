import gradio as gr
import random
import os
import json
import time
import modules.config
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.meta_parser
import utils.args_manager as args_manager
import utils.launch as launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json
from utils.consts import HOOOCUS_VERSION

def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    worker.async_tasks.append(task)

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
            if flag == 'results':
                ...
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


title = f'Hooocus {HOOOCUS_VERSION}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset


currentTask = worker.AsyncTask(args=[])

    

def stop_clicked(currentTask):
    import ldm_patched.modules.model_management as model_management
    currentTask.last_stop = 'stop'
    if (currentTask.processing):
        model_management.interrupt_current_processing()
    return currentTask

def skip_clicked(currentTask):
    import ldm_patched.modules.model_management as model_management
    currentTask.last_stop = 'skip'
    if (currentTask.processing):
        model_management.interrupt_current_processing()
    return currentTask


    



def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
    from extras.inpaint_mask import generate_mask_from_image

    extras = {}
    sam_options = None
    if mask_model == 'u2net_cloth_seg':
        extras['cloth_category'] = cloth_category
    elif mask_model == 'sam':
        sam_options = SAMOptions(
            dino_prompt=dino_prompt_text,
            dino_box_threshold=box_threshold,
            dino_text_threshold=text_threshold,
            dino_erode_or_dilate=dino_erode_or_dilate,
            dino_debug=dino_debug,
            max_detections=sam_max_detections,
            model_type=sam_model
        )

    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

    return mask



                

def trigger_metadata_preview(file):
    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

    results = {}
    if parameters is not None:
        results['parameters'] = parameters

    if isinstance(metadata_scheme, flags.MetadataScheme):
        results['metadata_scheme'] = metadata_scheme.value

    return results



def refresh_seed(r, seed_string):
    if r:
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed_value = int(seed_string)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                return seed_value
        except ValueError:
            pass
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)




def parse_meta(raw_prompt_txt, is_generating):
    loaded_json = None
    if is_json(raw_prompt_txt):
        loaded_json = json.loads(raw_prompt_txt)

    if loaded_json is None:
        if is_generating:
            return gr.update(), gr.update(), gr.update()
        else:
            return gr.update(), gr.update(visible=True), gr.update(visible=False)

    return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)




def trigger_metadata_import(file, state_is_generating):
    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
    if parameters is None:
        print('Could not find metadata in the image!')
        parsed_parameters = {}
    else:
        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
        parsed_parameters = metadata_parser.to_json(parameters)


def trigger_describe(modes, img, apply_styles):
    describe_prompts = []
    styles = set()

    if flags.describe_type_photo in modes:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        describe_prompts.append(default_interrogator_photo(img))
        styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

    if flags.describe_type_anime in modes:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        describe_prompts.append(default_interrogator_anime(img))
        styles.update(["Fooocus V2", "Fooocus Masterpiece"])

    if len(styles) == 0 or not apply_styles:
        styles = gr.update()
    else:
        styles = list(styles)

    if len(describe_prompts) == 0:
        describe_prompt = gr.update()
    else:
        describe_prompt = ', '.join(describe_prompts)

    return describe_prompt, styles

       



