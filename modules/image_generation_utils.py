import sys, os
from typing import List

from utils import config, flags
from utils.model_file_config import InpaintModelFiles

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import patch, meta_parser
from modules.private_logger import log
import math
import numpy as np
import time
import random
import copy
import cv2
import modules.default_pipeline as pipeline
from modules.core import encode_vae, numpy_to_pytorch
import utils.flags as flags
import modules.patch
import ldm_patched.modules.model_management
import extras.preprocessors as preprocessors
import extras.ip_adapter as ip_adapter
import extras.face_crop

from utils.model_file_config import UpscaleModel, InpaintModelFiles

from extras.censor import default_censor
from utils.sdxl_prompt_expansion_utils import (
    apply_style,
    get_random_style,
    apply_arrays,
    random_style_name,
)
from modules.private_logger import log
from extras.expansion import safe_str
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
from modules.upscaler import perform_upscale
from utils.logging_util import LoggingUtil
from utils.config import MAX_SEED


def progressbar(number, text):
    print(f"[Hooocus] {text}")
    return["preview", (number, text, None)]

def save_and_log(
        async_task: ImageGenerationSeed,
        height,
        imgs,
        task,
        use_expansion,
        width,
        loras,        
        persist_image=True,

        fooocus_expansion = None,
        pid = 0,
    ) -> list:
        img_paths = []
        for x in imgs:
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

            metadata_parser = None
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
            d.append(
                (
                    "Metadata Scheme",
                    "metadata_scheme",
                    (
                        METADATA_SCHEME
                        if async_task.save_metadata_to_images
                        else async_task.save_metadata_to_images
                    ),
                )
            )
            d.append(("Version", "version", "Hooocus v" + HOOOCUS_VERSION))
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
        modules.config.downloading_upscale_model()
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


def apply_image_input(
    async_task: config.ImageGenerationObject,
    base_model_additional_loras,
    clip_vision_path,
    controlnet_canny_path,
    controlnet_cpds_path,
    goals,
    inpaint_head_model_path,
    inpaint_image,
    inpaint_mask,
    inpaint_parameterized,
    ip_adapter_face_path,
    ip_adapter_path,
    ip_negative_path,
    skip_prompt_processing,
    use_synthetic_refiner,
):

    if (
        (
            async_task.use_upscale_or_vary
            or (
                async_task.use_imageprompt
                and async_task.mix_image_prompt_and_vary_upscale
            )
        )
        and async_task.uov_method and async_task.uov_input_image is not None
    ):
        async_task.uov_input_image, skip_prompt_processing, async_task.steps = (
            prepare_upscale(
                async_task,
                goals,
                async_task.uov_input_image,
                async_task.uov_method,
                async_task.performance_selection,
                async_task.steps,
                1,
                skip_prompt_processing=skip_prompt_processing,
            )
        )
    if (
        async_task.inpaint_options or (async_task.use_imageprompt and async_task.mix_image_prompt_and_inpaint)):
        
        inpaint_options = async_task.inpaint_options
        inpaint_image = inpaint_options.inpaint_input_image["image"]
        inpaint_mask = inpaint_options.inpaint_input_image["mask"][:, :, 0]

        if inpaint_options.use_advanced_inpaint_masking:
            if inpaint_options.inpaint_mask_image_upload:
                image_mask_uploaded = np.maximum(
                    inpaint_options.inpaint_mask_image_upload["image"],
                    inpaint_options.inpaint_mask_image_upload["mask"])
                
                H, W, C = inpaint_image.shape
                image_mask_uploaded = resample_image(
                    image_mask_uploaded, width=W, height=H
                )
                image_mask_uploaded = np.mean(
                    image_mask_uploaded, axis=2
                )
                image_mask_uploaded = (
                    image_mask_uploaded > 127
                ).astype(np.uint8) * 255
                inpaint_mask = np.maximum(inpaint_mask, image_mask_uploaded)

        if int(inpaint_options.inpaint_erode_or_dilate) != 0:
            inpaint_mask = erode_or_dilate(
                inpaint_mask, inpaint_options.inpaint_erode_or_dilate
            )

        if inpaint_options.invert_mask:
            inpaint_mask = 255 - inpaint_mask

        inpaint_image = ensure_three_channels(inpaint_image)
        if (np.any(inpaint_mask > 127) or len(inpaint_options.outpaint_selections) > 0):
            progressbar(async_task, 1, "Downloading upscale models ...")
            UpscaleModel.download_model()
            if inpaint_parameterized:
                progressbar(async_task, 1, "Downloading inpainter ...")
                inpaint_head_model_path = InpaintModelFiles.InpaintHead.download_model()
                inpaint_patch_model_path = (InpaintModelFiles.InpaintPatchV26.download_model())
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                
                print(f"[Inpaint] Current inpaint model is {inpaint_patch_model_path}")
                
                if not async_task.refiner_model:
                    use_synthetic_refiner = True
                    async_task.refiner_switch = 0.8
            else:
                inpaint_head_model_path, inpaint_patch_model_path = None, None
                print(f"[Inpaint] Parameterized inpaint is disabled.")

            if inpaint_options.inpaint_additional_prompt:
                if async_task.prompt == "":
                    async_task.prompt = inpaint_options.inpaint_additional_prompt
                else:
                    async_task.prompt = (
                        inpaint_options.inpaint_additional_prompt
                        + "\n"
                        + async_task.prompt
                    )
            async_task.goals.append("inpaint")
    if (
        async_task.current_tab == "ip"
        or async_task.mixing_image_prompt_and_vary_upscale
        or async_task.mixing_image_prompt_and_inpaint
    ):
        goals.append("cn")
        progressbar(async_task, 1, "Downloading control models ...")
        if len(async_task.cn_tasks[flags.cn_canny]) > 0:
            controlnet_canny_path = downloading_controlnet_canny()
        if len(async_task.cn_tasks[flags.cn_cpds]) > 0:
            controlnet_cpds_path = config.downloading_controlnet_cpds()
        if len(async_task.cn_tasks[flags.cn_ip]) > 0:
            clip_vision_path, ip_negative_path, ip_adapter_path = (
                config.downloading_ip_adapters("ip")
            )
        if len(async_task.cn_tasks[flags.cn_ip_face]) > 0:
            clip_vision_path, ip_negative_path, ip_adapter_face_path = (
                config.downloading_ip_adapters("face")
            )
    if (
        async_task.current_tab == "enhance"
        and async_task.enhance_input_image is not None
    ):
        goals.append("enhance")
        skip_prompt_processing = True
        async_task.enhance_input_image = ensure_three_channels(async_task.enhance_input_image)
    return (
        base_model_additional_loras,
        clip_vision_path,
        controlnet_canny_path,
        controlnet_cpds_path,
        inpaint_head_model_path,
        inpaint_image,
        inpaint_mask,
        ip_adapter_face_path,
        ip_adapter_path,
        ip_negative_path,
        skip_prompt_processing,
        use_synthetic_refiner,
    )



def apply_freeu(async_task):
    print(f"FreeU is enabled!")
    pipeline.final_unet = core.apply_freeu(
        pipeline.final_unet,
        async_task.freeu_b1,
        async_task.freeu_b2,
        async_task.freeu_s1,
        async_task.freeu_s2,
    )

def patch_discrete(unet, scheduler_name):
    return core.opModelSamplingDiscrete.patch(unet, scheduler_name, False)[0]

def patch_edm(unet, scheduler_name):
    return core.opModelSamplingContinuousEDM.patch(
        unet, scheduler_name, 120.0, 0.002
    )[0]

def patch_samplers(async_task):
    final_scheduler_name = async_task.scheduler_name

    if async_task.scheduler_name in ["lcm", "tcd"]:
        final_scheduler_name = "sgm_uniform"
        if pipeline.final_unet is not None:
            pipeline.final_unet = patch_discrete(
                pipeline.final_unet, async_task.scheduler_name
            )
        if pipeline.final_refiner_unet is not None:
            pipeline.final_refiner_unet = patch_discrete(
                pipeline.final_refiner_unet, async_task.scheduler_name
            )

    elif async_task.scheduler_name == "edm_playground_v2.5":
        final_scheduler_name = "karras"
        if pipeline.final_unet is not None:
            pipeline.final_unet = patch_edm(
                pipeline.final_unet, async_task.scheduler_name
            )
        if pipeline.final_refiner_unet is not None:
            pipeline.final_refiner_unet = patch_edm(
                pipeline.final_refiner_unet, async_task.scheduler_name
            )

    return final_scheduler_name






def apply_overrides(async_task, steps, height, width):
    if async_task.overwrite_step > 0:
        steps = async_task.overwrite_step
    switch = int(round(async_task.steps * async_task.refiner_switch))
    if async_task.overwrite_switch > 0:
        switch = async_task.overwrite_switch
    if async_task.overwrite_width > 0:
        width = async_task.overwrite_width
    if async_task.overwrite_height > 0:
        height = async_task.overwrite_height
    return steps, switch, width, height



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
            if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
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
            yield_result(
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
            modules.config.downloading_inpaint_models(inpaint_engine)
        )
        if inpaint_patch_model_path not in base_model_additional_loras:
            base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
    progressbar(async_task, current_progress, "Preparing enhance prompts ...")
    # positive and negative conditioning aren't available here anymore, process prompt again
    tasks_enhance, use_expansion, loras, current_progress = process_prompt(
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
    imgs, img_paths, current_progress = process_task(
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
    inpaint_worker.current_task = None

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






                        

def apply_control_nets(
    async_task: ImageGenerationSeed,
    height,
    ip_adapter_face_path,
    ip_adapter_path,
    width,
    current_progress,
):
    for task in async_task.cn_tasks[flags.cn_canny]:
        cn_img = resize_image(ensure_three_channels(cn_img), width=width, height=height)

        if not async_task.skipping_cn_preprocessor:
            cn_img = preprocessors.canny_pyramid(
                cn_img,
                async_task.canny_low_threshold,
                async_task.canny_high_threshold,
            )

        cn_img = ensure_three_channels(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)
        if async_task.debugging_cn_preprocessor:
            yield_result(
                async_task,
                cn_img,
                current_progress,
                async_task.black_out_nsfw,
                do_not_show_finished_images=True,
            )
    for task in async_task.cn_tasks[flags.cn_cpds]:
        cn_img, cn_stop, cn_weight = task
        cn_img = resize_image(ensure_three_channels(cn_img), width=width, height=height)

        if not async_task.skipping_cn_preprocessor:
            cn_img = preprocessors.cpds(cn_img)

        cn_img = ensure_three_channels(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)
        if async_task.debugging_cn_preprocessor:
            yield_result(
                async_task,
                cn_img,
                current_progress,
                async_task.black_out_nsfw,
                do_not_show_finished_images=True,
            )
    for task in async_task.cn_tasks[flags.cn_ip]:
        cn_img, cn_stop, cn_weight = task
        cn_img = ensure_three_channels(cn_img)

        # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

        task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
        if async_task.debugging_cn_preprocessor:
            yield_result(
                async_task,
                cn_img,
                current_progress,
                async_task.black_out_nsfw,
                do_not_show_finished_images=True,
            )
    for task in async_task.cn_tasks[flags.cn_ip_face]:
        cn_img, cn_stop, cn_weight = task
        cn_img = ensure_three_channels(cn_img)

        if not async_task.skipping_cn_preprocessor:
            cn_img = extras.face_crop.crop_image(cn_img)

        # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

        task[0] = ip_adapter.preprocess(
            cn_img, ip_adapter_path=ip_adapter_face_path
        )
        if async_task.debugging_cn_preprocessor:
            yield_result(
                async_task,
                cn_img,
                current_progress,
                async_task.black_out_nsfw,
                do_not_show_finished_images=True,
            )
    all_ip_tasks = (
        async_task.cn_tasks[flags.cn_ip] + async_task.cn_tasks[flags.cn_ip_face]
    )
    if len(all_ip_tasks) > 0:
        pipeline.final_unet = ip_adapter.patch_model(
            pipeline.final_unet, all_ip_tasks
        )

def apply_vary(
    async_task: ImageGenerationSeed,
    uov_method,
    denoising_strength,
    uov_input_image,
    switch,
    current_progress,
    advance_progress=False,
):
    if "subtle" in uov_method:
        denoising_strength = 0.5
    if "strong" in uov_method:
        denoising_strength = 0.85
    if async_task.overwrite_vary_strength > 0:
        denoising_strength = async_task.overwrite_vary_strength
    shape_ceil = get_image_shape_ceil(uov_input_image)
    if shape_ceil < 1024:
        print(f"[Vary] Image is resized because it is too small.")
        shape_ceil = 1024
    elif shape_ceil > 2048:
        print(f"[Vary] Image is resized because it is too big.")
        shape_ceil = 2048
    uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)
    initial_pixels = core.numpy_to_pytorch(uov_input_image)
    if advance_progress:
        current_progress += 1
    progressbar(async_task, current_progress, "VAE encoding ...")
    candidate_vae, _ = pipeline.get_candidate_vae(
        steps=async_task.steps,
        switch=switch,
        denoise=denoising_strength,
        refiner_swap_method=async_task.refiner_swap_method,
    )
    initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
    B, C, H, W = initial_latent["samples"].shape
    width = W * 8
    height = H * 8
    print(f"Final resolution is {str((width, height))}.")
    return (
        uov_input_image,
        denoising_strength,
        initial_latent,
        width,
        height,
        current_progress,
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

    async_task.current_task = inpaint_worker.InpaintWorker(
        image=inpaint_image,
        mask=inpaint_mask,
        use_fill=denoising_strength > 0.99,
        k=inpaint_respective_field,
    )
    if async_task.debugging_inpaint_preprocessor:
        yield_result(
            async_task,
            inpaint_worker.current_task.visualize_mask_processing(),
            100,
            async_task.black_out_nsfw,
            do_not_show_finished_images=True,
        )
        raise EarlyReturnException


    inpaint_pixel_fill = core.numpy_to_pytorch(
        inpaint_worker.current_task.interested_fill
    )
    inpaint_pixel_image = core.numpy_to_pytorch(
        inpaint_worker.current_task.interested_image
    )
    inpaint_pixel_mask = core.numpy_to_pytorch(
        inpaint_worker.current_task.interested_mask
    )
    
    candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
        steps=async_task.steps,
        switch=switch,
        denoise=denoising_strength,
        refiner_swap_method=async_task.refiner_swap_method,
    )
    latent_inpaint, latent_mask = core.encode_vae_inpaint(
        mask=inpaint_pixel_mask, vae=candidate_vae, pixels=inpaint_pixel_image
    )
    
    latent_swap = None
    if candidate_vae_swap is not None:
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, "VAE SD15 encoding ...")
        latent_swap = core.encode_vae(
            vae=candidate_vae_swap, pixels=inpaint_pixel_fill
        )["samples"]
    if advance_progress:
        current_progress += 1
    progressbar(async_task, current_progress, "VAE encoding ...")
    latent_fill = core.encode_vae(vae=candidate_vae, pixels=inpaint_pixel_fill)[
        "samples"
    ]
    inpaint_worker.current_task.load_latent(
        latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap
    )
    if inpaint_parameterized:
        pipeline.final_unet = inpaint_worker.current_task.patch(
            inpaint_head_model_path=inpaint_head_model_path,
            inpaint_latent=latent_inpaint,
            inpaint_latent_mask=latent_mask,
            model=pipeline.final_unet,
        )
    if not inpaint_disable_initial_latent:
        initial_latent = {"samples": latent_fill}
    B, C, H, W = latent_fill.shape
    height, width = H * 8, W * 8
    final_height, final_width = inpaint_worker.current_task.image.shape[:2]
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

def apply_upscale(
    uov_input_image,
    uov_method,
    candidate_vae,
    overwrite_upscale_strength):

    H, W, C = uov_input_image.shape
    uov_input_image = perform_upscale(uov_input_image)
    
    print(f"Image upscaled.")
    if "1.5x" in uov_method:
        f = 1.5
    elif "2x" in uov_method:
        f = 2.0
    else:
        f = 1.0
    shape_ceil = get_shape_ceil(H * f, W * f)
    if shape_ceil < 1024:
        print(f"[Upscale] Image is resized because it is too small.")
        uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
        shape_ceil = 1024
    else:
        uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)
    image_is_super_large = shape_ceil > 2800
    if "fast" in uov_method:
        direct_return = True
    elif image_is_super_large:
        print(
            "Image is too large. Directly returned the SR image. "
            "Usually directly return SR image at 4K resolution "
            "yields better results than SDXL diffusion."
        )
        direct_return = True
    else:
        direct_return = False
    if direct_return:
        return (
            direct_return,
            uov_input_image,
            None,
            None,
            None,
            None,
            None
        )

    tiled = True
    
    if overwrite_upscale_strength > 0:
        denoising_strength = overwrite_upscale_strength
    
    initial_pixels = numpy_to_pytorch(uov_input_image)

    initial_latent = encode_vae(
        vae=candidate_vae, pixels=initial_pixels, tiled=True
    )
    B, C, H, W = initial_latent["samples"].shape
    width = W * 8
    height = H * 8
    print(f"Final resolution is {str((width, height))}.")
    
    return (
        direct_return,
        uov_input_image,
        denoising_strength,
        initial_latent,
        tiled,
        width,
        height,
    )


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
