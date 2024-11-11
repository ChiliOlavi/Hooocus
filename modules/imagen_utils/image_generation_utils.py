import os
import sys
import time
import math
import random
import copy
from typing import List

import numpy as np
import cv2

from modules.patch_modules import patch
from utils import config, flags
from utils.model_file_config import InpaintModelFiles, UpscaleModel
from utils.sdxl_prompt_expansion_utils import (
    apply_style,
    get_random_style,
    apply_arrays,
    random_style_name,
)
from utils.logging_util import LoggingUtil
from utils.config import MAX_SEED, ImageGenerationObject, ImageGenerationSeed, PatchSettings

from modules import default_pipeline as pipeline
from modules.imagen_utils.private_logger import log
from modules.core import encode_vae, numpy_to_pytorch
from modules.util import (
    remove_empty_str,
    ensure_three_channels,
    resize_image,
    get_image_shape_ceil,
    set_image_shape_ceil,
    get_shape_ceil,
    resample_image,
    erode_or_dilate,
    parse_lora_references_from_prompt,
    apply_wildcards,
)
from modules.imagen_utils.upscale.upscaler import perform_upscale

import ldm_patched.modules.model_management
import extras.preprocessors as preprocessors
import extras.ip_adapter as ip_adapter
import extras.face_crop
from extras.censor import default_censor
from extras.expansion import safe_str

from unavoided_global_vars import patch_settings_GLOBAL_CAUTION, opModelSamplingContinuousEDM
from wasteland import meta_parser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def log(
    async_task: ImageGenerationSeed,
    height: int,
    imgs: List[np.ndarray],
    task: dict,
    use_expansion: bool,
    width: int,
    loras: List[tuple],
    persist_image: bool = True,
    fooocus_expansion: str = None,
    pid: int = 0,
) -> List[str]:
    """Save images and log metadata.

    Args:
        async_task (ImageGenerationSeed): The asynchronous task object.
        height (int): The height of the image.
        imgs (List[np.ndarray]): List of images to save.
        task (dict): Task details.
        use_expansion (bool): Whether to use expansion.
        width (int): The width of the image.
        loras (List[tuple]): List of LoRA configurations.
        persist_image (bool, optional): Whether to persist the image. Defaults to True.
        fooocus_expansion (str, optional): Fooocus expansion details. Defaults to None.
        pid (int, optional): Process ID. Defaults to 0.

    Returns:
        List[str]: List of image paths.
    """
    img_paths = []
    for x in imgs:
        d = generate_metadata_dict(async_task, task, use_expansion, fooocus_expansion, width, height, loras, pid)
        metadata_parser = create_metadata_parser(async_task, task, loras)
        img_paths.append(
            log(
                x,
                async_task.path_outputs,
                d,
                metadata_parser,
                async_task.output_format,
                task,
                persist_image,
            )
        )
    return img_paths

def generate_metadata_dict(async_task, task, use_expansion, fooocus_expansion, width, height, loras, pid):
    """Generate metadata dictionary for logging."""
    d = [
        ("Prompt", "prompt", task["log_positive_prompt"]),
        ("Negative Prompt", "negative_prompt", task["log_negative_prompt"]),
        ("Fooocus V2 Expansion", "prompt_expansion", task["expansion"]),
        (
            "Styles",
            "styles",
            str(
                task["styles"]
                if not use_expansion
                else [fooocus_expansion] + task["styles"]
            ),
        ),
        ("Performance", "performance", async_task.performance_selection.value),
        ("Steps", "steps", async_task.steps),
        ("Resolution", "resolution", str((width, height))),
        ("Guidance Scale", "guidance_scale", async_task.cfg_scale),
        ("Sharpness", "sharpness", async_task.sharpness),
        (
            "ADM Guidance",
            "adm_guidance",
            str(
                (
                    patch.patch_settings[pid].positive_adm_scale,
                    patch.patch_settings[pid].negative_adm_scale,
                    patch.patch_settings[pid].adm_scaler_end,
                )
            ),
        ),
        ("Base Model", "base_model", async_task.base_model_name),
        ("Refiner Model", "refiner_model", async_task.refiner_model_name),
        ("Refiner Switch", "refiner_switch", async_task.refiner_switch),
    ]

    if async_task.refiner_model_name != "None":
        if async_task.overwrite_switch > 0:
            d.append(
                (
                    "Overwrite Switch",
                    "overwrite_switch",
                    async_task.overwrite_switch,
                )
            )
        if async_task.refiner_swap_method != flags.refiner_swap_method:
            d.append(
                (
                    "Refiner Swap Method",
                    "refiner_swap_method",
                    async_task.refiner_swap_method,
                )
            )
    if (
        patch.patch_settings[pid].adaptive_cfg
        != config.default_cfg_tsnr
    ):
        d.append(
            (
                "CFG Mimicking from TSNR",
                "adaptive_cfg",
                patch.patch_settings[pid].adaptive_cfg,
            )
        )

    if async_task.clip_skip > 1:
        d.append(("CLIP Skip", "clip_skip", async_task.clip_skip))
    d.append(("Sampler", "sampler", async_task.sampler_name))
    d.append(("Scheduler", "scheduler", async_task.scheduler_name))
    d.append(("VAE", "vae", async_task.vae_name))
    d.append(("Seed", "seed", str(task["task_seed"])))

    if async_task.freeu_enabled:
        d.append(
            (
                "FreeU",
                "freeu",
                str(
                    (
                        async_task.freeu_b1,
                        async_task.freeu_b2,
                        async_task.freeu_s1,
                        async_task.freeu_s2,
                    )
                ),
            )
        )

    for li, (n, w) in enumerate(loras):
        if n != "None":
            d.append(
                (f"LoRA {li + 1}", f"lora_combined_{li + 1}", f"{n} : {w}")
            )
    return d

def create_metadata_parser(async_task, task, loras):
    """Create a metadata parser if needed."""
    if async_task.save_metadata_to_images:
        metadata_parser = meta_parser.HooocusMetadataParser()
        metadata_parser.set_data(
            task["log_positive_prompt"],
            task["positive"],
            task["log_negative_prompt"],
            task["negative"],
            async_task.steps,
            async_task.base_model_name,
            async_task.refiner_model_name,
            loras,
            async_task.vae_name,
        )
        return metadata_parser
    return None

def prepare_upscale(
    async_task: ImageGenerationSeed,
    goals,
    uov_input_image,
    uov_method,
    performance,
    steps,
    current_progress,
    advance_progress=False,
    skip_prompt_processing=False,
):
    uov_input_image = ensure_three_channels(uov_input_image)
    
    if "vary" in uov_method:
        goals.append("vary")
    elif "upscale" in uov_method:
        goals.append("upscale")
        if "fast" in uov_method:
            skip_prompt_processing = True
            steps = 0
        else:
            steps = performance.steps_uov()

        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, "Downloading upscale models ...")
        config.downloading_upscale_model()
    return uov_input_image, skip_prompt_processing, steps

def prepare_enhance_prompt(prompt: str, fallback_prompt: str):
    if (
        safe_str(prompt) == ""
        or len(
            remove_empty_str([safe_str(p) for p in prompt.splitlines()], default="")
        )
        == 0
    ):
        prompt = fallback_prompt

    return prompt

















def stop_processing(async_task, processing_start_time):
    async_task.processing = False
    processing_time = time.perf_counter() - processing_start_time
    print(f"Processing time (total): {processing_time:.2f} seconds")

def process_enhance(
    all_steps,
    async_task: ImageGenerationSeed,
    callback,
    controlnet_canny_path,
    controlnet_cpds_path,
    current_progress,
    current_task_id,
    denoising_strength,
    inpaint_disable_initial_latent,
    inpaint_engine,
    inpaint_respective_field,
    inpaint_strength,
    prompt,
    negative_prompt,
    final_scheduler_name,
    goals,
    height,
    img,
    mask,
    preparation_steps,
    steps,
    switch,
    tiled,
    total_count,
    use_expansion,
    use_style,
    use_synthetic_refiner,
    width,
    show_intermediate_results=True,
    persist_image=True,
):
    base_model_additional_loras = []
    inpaint_head_model_path = None
    inpaint_parameterized = (
        inpaint_engine != "None"
    )  # inpaint_engine = None, improve detail
    initial_latent = None

    prompt = prepare_enhance_prompt(prompt, async_task.prompt)
    negative_prompt = prepare_enhance_prompt(
        negative_prompt, async_task.negative_prompt
    )

    if "vary" in goals:
        img, denoising_strength, initial_latent, width, height, current_progress = (
            apply_vary(
                async_task,
                async_task.enhance_uov_method,
                denoising_strength,
                img,
                switch,
                current_progress,
            )
        )
    if "upscale" in goals:
        (
            direct_return,
            img,
            denoising_strength,
            initial_latent,
            tiled,
            width,
            height,
            current_progress,
        ) = apply_upscale(
            async_task, img, async_task.enhance_uov_method, switch, current_progress
        )
        if direct_return:
            d = [("Upscale (Fast)", "upscale_fast", "2x")]
            if config.default_black_out_nsfw or async_task.black_out_nsfw:
                progressbar(
                    async_task, current_progress, "Checking for NSFW content ..."
                )
                img = default_censor(img)
            progressbar(
                async_task,
                current_progress,
                f"Saving image {current_task_id + 1}/{total_count} to system ...",
            )
            uov_image_path = log(
                img,
                async_task.path_outputs,
                d,
                output_format=async_task.output_format,
                persist_image=persist_image,
            )
            pipeline.yield_result(
                async_task,
                uov_image_path,
                current_progress,
                async_task.black_out_nsfw,
                False,
                do_not_show_finished_images=not show_intermediate_results
                or async_task.disable_intermediate_results,
            )
            return current_progress, img, prompt, negative_prompt

    if "inpaint" in goals and inpaint_parameterized:
        progressbar(async_task, current_progress, "Downloading inpainter ...")
        inpaint_head_model_path, inpaint_patch_model_path = (
            config.downloading_inpaint_models(inpaint_engine)
        )
        if inpaint_patch_model_path not in base_model_additional_loras:
            base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
    progressbar(async_task, current_progress, "Preparing enhance prompts ...")
    # positive and negative conditioning aren't available here anymore, process prompt again
    tasks_enhance, use_expansion, loras, current_progress = pipeline.process_prompt(
        async_task,
        prompt,
        negative_prompt,
        base_model_additional_loras,
        1,
        True,
        use_expansion,
        use_style,
        use_synthetic_refiner,
        current_progress,
    )
    task_enhance = tasks_enhance[0]
    # TODO could support vary, upscale and CN in the future
    # if 'cn' in goals:
    #     apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width)
    if async_task.freeu_enabled:
        apply_freeu(async_task)
    patch_samplers(async_task)
    if "inpaint" in goals:
        denoising_strength, initial_latent, width, height, current_progress = (
            apply_inpaint(
                async_task,
                None,
                inpaint_head_model_path,
                img,
                mask,
                inpaint_parameterized,
                inpaint_strength,
                inpaint_respective_field,
                switch,
                inpaint_disable_initial_latent,
                current_progress,
                True,
            )
        )
    imgs, img_paths, current_progress = pipeline.process_task(
        all_steps,
        async_task,
        callback,
        controlnet_canny_path,
        controlnet_cpds_path,
        current_task_id,
        denoising_strength,
        final_scheduler_name,
        goals,
        initial_latent,
        steps,
        switch,
        task_enhance["c"],
        task_enhance["uc"],
        task_enhance,
        loras,
        tiled,
        use_expansion,
        width,
        height,
        current_progress,
        preparation_steps,
        total_count,
        show_intermediate_results,
        persist_image,
    )

    del task_enhance["c"], task_enhance["uc"]  # Save memory
    return current_progress, imgs[0], prompt, negative_prompt

def enhance_upscale(
    all_steps,
    async_task,
    base_progress,
    callback,
    controlnet_canny_path,
    controlnet_cpds_path,
    current_task_id,
    denoising_strength,
    done_steps_inpainting,
    done_steps_upscaling,
    enhance_steps,
    prompt,
    negative_prompt,
    final_scheduler_name,
    height,
    img,
    preparation_steps,
    switch,
    tiled,
    total_count,
    use_expansion,
    use_style,
    use_synthetic_refiner,
    width,
    persist_image=True,
):
    # reset inpaint worker to prevent tensor size issues and not mix upscale and inpainting
    inpaint_worker = None

    current_progress = int(
        base_progress
        + (100 - preparation_steps)
        / float(all_steps)
        * (done_steps_upscaling + done_steps_inpainting)
    )
    goals_enhance = []
    img, skip_prompt_processing, steps = prepare_upscale(
        async_task,
        goals_enhance,
        img,
        async_task.enhance_uov_method,
        async_task.performance_selection,
        enhance_steps,
        current_progress,
    )
    steps, _, _, _ = apply_overrides(async_task, steps, height, width)
    exception_result = ""
    if len(goals_enhance) > 0:
        try:
            current_progress, img, prompt, negative_prompt = process_enhance(
                all_steps,
                async_task,
                callback,
                controlnet_canny_path,
                controlnet_cpds_path,
                current_progress,
                current_task_id,
                denoising_strength,
                False,
                "None",
                0.0,
                0.0,
                prompt,
                negative_prompt,
                final_scheduler_name,
                goals_enhance,
                height,
                img,
                None,
                preparation_steps,
                steps,
                switch,
                tiled,
                total_count,
                use_expansion,
                use_style,
                use_synthetic_refiner,
                width,
                persist_image=persist_image,
            )

        except ldm_patched.modules.model_management.InterruptProcessingException:
            if async_task.last_stop == "skip":
                print("User skipped")
                async_task.last_stop = False
                # also skip all enhance steps for this image, but add the steps to the progress bar
                if (
                    async_task.enhance_uov_processing_order
                    == flags.enhancement_uov_before
                ):
                    done_steps_inpainting += (
                        len(async_task.enhance_ctrls) * enhance_steps
                    )
                exception_result = "continue"
            else:
                print("User stopped")
                exception_result = "break"
        finally:
            done_steps_upscaling += steps
    return (
        current_task_id,
        done_steps_inpainting,
        done_steps_upscaling,
        img,
        exception_result,
    )






                        



def apply_inpaint(
    async_task,
    initial_latent,
    inpaint_head_model_path,
    inpaint_image,
    inpaint_mask,
    inpaint_parameterized,
    denoising_strength,
    inpaint_respective_field,
    switch,
    inpaint_disable_initial_latent,
    current_progress,
    skip_apply_outpaint=False,
    advance_progress=False,
):
    if not skip_apply_outpaint:
        inpaint_image, inpaint_mask = apply_outpaint(
            async_task, inpaint_image, inpaint_mask
        )

    async_task.current_task = pipeline.inpaint_worker.InpaintWorker(
        image=inpaint_image,
        mask=inpaint_mask,
        use_fill=denoising_strength > 0.99,
        k=inpaint_respective_field,
    )
    if async_task.debugging_inpaint_preprocessor:
        pipeline.yield_result(
            async_task,
            async_task.current_task.visualize_mask_processing(),
            100,
            async_task.black_out_nsfw,
            do_not_show_finished_images=True,
        )
        raise pipeline.EarlyReturnException


    inpaint_pixel_fill = pipeline.core.numpy_to_pytorch(
        async_task.current_task.interested_fill
    )
    inpaint_pixel_image = pipeline.core.numpy_to_pytorch(
        async_task.current_task.interested_image
    )
    inpaint_pixel_mask = pipeline.core.numpy_to_pytorch(
        async_task.current_task.interested_mask
    )
    
    candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
        steps=async_task.steps,
        switch=switch,
        denoise=denoising_strength,
        refiner_swap_method=async_task.refiner_swap_method,
    )
    latent_inpaint, latent_mask = pipeline.core.encode_vae_inpaint(
        mask=inpaint_pixel_mask, vae=candidate_vae, pixels=inpaint_pixel_image
    )
    
    latent_swap = None
    if candidate_vae_swap is not None:
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, "VAE SD15 encoding ...")
        latent_swap = pipeline.core.encode_vae(
            vae=candidate_vae_swap, pixels=inpaint_pixel_fill
        )["samples"]
    if advance_progress:
        current_progress += 1
    progressbar(async_task, current_progress, "VAE encoding ...")
    latent_fill = pipeline.core.encode_vae(vae=candidate_vae, pixels=inpaint_pixel_fill)[
        "samples"
    ]
    async_task.current_task.load_latent(
        latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap
    )
    if inpaint_parameterized:
        pipeline.final_unet = async_task.current_task.patch(
            inpaint_head_model_path=inpaint_head_model_path,
            inpaint_latent=latent_inpaint,
            inpaint_latent_mask=latent_mask,
            model=pipeline.final_unet,
        )
    if not inpaint_disable_initial_latent:
        initial_latent = {"samples": latent_fill}
    B, C, H, W = latent_fill.shape
    height, width = H * 8, W * 8
    final_height, final_width = async_task.current_task.image.shape[:2]
    print(
        f"Final resolution is {str((final_width, final_height))}, latent is {str((width, height))}."
    )

    return denoising_strength, initial_latent, width, height, current_progress

def apply_outpaint(async_task, inpaint_image, inpaint_mask):
    if len(async_task.outpaint_selections) > 0:
        H, W, C = inpaint_image.shape
        if "top" in async_task.outpaint_selections:
            inpaint_image = np.pad(
                inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode="edge"
            )
            inpaint_mask = np.pad(
                inpaint_mask,
                [[int(H * 0.3), 0], [0, 0]],
                mode="constant",
                constant_values=255,
            )
        if "bottom" in async_task.outpaint_selections:
            inpaint_image = np.pad(
                inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode="edge"
            )
            inpaint_mask = np.pad(
                inpaint_mask,
                [[0, int(H * 0.3)], [0, 0]],
                mode="constant",
                constant_values=255,
            )

        H, W, C = inpaint_image.shape
        if "left" in async_task.outpaint_selections:
            inpaint_image = np.pad(
                inpaint_image, [[0, 0], [int(W * 0.3), 0], [0, 0]], mode="edge"
            )
            inpaint_mask = np.pad(
                inpaint_mask,
                [[0, 0], [int(W * 0.3), 0]],
                mode="constant",
                constant_values=255,
            )
        if "right" in async_task.outpaint_selections:
            inpaint_image = np.pad(
                inpaint_image, [[0, 0], [0, int(W * 0.3)], [0, 0]], mode="edge"
            )
            inpaint_mask = np.pad(
                inpaint_mask,
                [[0, 0], [0, int(W * 0.3)]],
                mode="constant",
                constant_values=255,
            )

        inpaint_image = np.ascontiguousarray(inpaint_image.copy())
        inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
        async_task.inpaint_strength = 1.0
        async_task.inpaint_respective_field = 1.0
    return inpaint_image, inpaint_mask

def build_image_wall(image_paths: List[str]) -> List[np.ndarray] | None:
    """Builds a wall of images from a list of image paths.
    
    ### Args
    - image_paths (List[str]): List of image paths.

    ### Returns
    - List[np.ndarray] | None: List of images in the wall.
    """
    results: List[np.ndarray] = []

    if len(image_paths) < 2:
        return

    for img in image_paths:
        if os.path.exists(img):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img.ndim != 3:
            return
        
        results.append(img)

    H, W, C = results[0].shape

    for img in results:
        Hn, Wn, Cn = img.shape
        if H != Hn or C != Cn or W != Wn:
            return None

    cols = float(len(results)) ** 0.5
    cols = int(math.ceil(cols))
    rows = float(len(results)) / float(cols)
    rows = int(math.ceil(rows))

    wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            if y * cols + x < len(results):
                img = results[y * cols + x]
                wall[y * H : y * H + H, x * W : x * W + W, :] = img

    return [wall]
