# OLD CODE START ###

import os
import sys
import threading

from pydantic import BaseModel

from modules.image_generation_utils import (
    apply_control_nets,
    apply_freeu,
    apply_image_input,
    apply_inpaint,
    apply_overrides,
    apply_patch_settings,
    apply_upscale,
    apply_vary,
    build_image_wall,
    enhance_upscale,
    patch_samplers,
    process_prompt,
    save_and_log,
    stop_processing,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.data_models import ControlNetTask
import utils.config as config
from utils.logging_util import LoggingUtil

from extras.inpaint_mask import generate_mask_from_image, SAMOptions
from modules.patch import PatchSettings, patch_all

from utils.hooocus_utils import ImageGenerationSeed

patch_all()

GlobalConfig = config.GLOBAL_CONFIG
logger = LoggingUtil().get_logger()

async_tasks: list[ImageGenerationSeed] = []

class EarlyReturnException(BaseException):
    pass

import modules.default_pipeline as pipeline
def worker():
    global async_tasks

    import os
    import traceback
    import torch
    import time
    import modules.default_pipeline as pipeline
    import modules.core as core
    import utils.flags as flags
    import modules.patch
    import ldm_patched.modules.model_management
    import modules.inpaint_worker as inpaint_worker
    import extras.ip_adapter as ip_adapter

    from extras.censor import default_censor
    from modules.sdxl_styles import fooocus_expansion
    from modules.private_logger import log
    from modules.util import erode_or_dilate
    from utils.flags import Performance
    from modules.image_generation_utils import progressbar

    pid = os.getpid()
    print(f"Started worker with PID {pid}")

    def yield_result(
        async_task: ImageGenerationSeed,
        imgs,
        progressbar_index,
        do_not_show_finished_images=False,
    ):

        if (GlobalConfig.default_black_out_nsfw):
            progressbar(async_task, progressbar_index, "Checking for NSFW content ...")
            imgs = default_censor(imgs)

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        async_task.yields.append(["results", async_task.results])
        return

    def process_task(
        all_steps,
        async_task: ImageGenerationSeed,
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
        positive_cond,
        negative_cond,
        task,
        loras,
        tiled,
        use_expansion,
        width,
        height,
        base_progress,
        preparation_steps,
        total_count,
        show_intermediate_results,
        persist_image=True,
    ):
        if async_task.last_stop is not False:
            ldm_patched.modules.model_management.interrupt_current_processing()

        if async_task.cn_tasks:
            for cn_flag, cn_path in [
                (flags.cn_canny, controlnet_canny_path),
                (flags.cn_cpds, controlnet_cpds_path),
            ]:

                controlnet_task: ControlNetTask
                for controlnet_task in async_task.cn_tasks:
                    if controlnet_task.cn_type == cn_flag:
                        positive_cond, negative_cond = core.apply_controlnet(
                            positive_cond,
                            negative_cond,
                            pipeline.loaded_ControlNets[cn_path],
                            controlnet_task.cn_img,
                            controlnet_task.cn_weight,
                            0,
                            controlnet_task.cn_stop,
                        )

        imgs = pipeline.process_diffusion(
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            steps=steps,
            switch=switch,
            width=width,
            height=height,
            image_seed=task["task_seed"],
            callback=callback,
            sampler_name=async_task.sampler_name,
            scheduler_name=final_scheduler_name,
            latent=initial_latent,
            denoise=denoising_strength,
            tiled=tiled,
            cfg_scale=async_task.cfg_scale,
            refiner_swap_method=async_task.refiner_swap_method,
            disable_preview=async_task.disable_preview,
        )

        del positive_cond, negative_cond  # Save memory
        if inpaint_worker.current_task is not None:
            imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

        current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * steps)

        progressbar(async_task, current_progress, f"Saving image {current_task_id + 1}/{total_count} to system ...")

        img_paths = save_and_log(
            async_task,
            height,
            imgs,
            task,
            use_expansion,
            width,
            loras,
            persist_image,
            fooocus_expansion=fooocus_expansion,
            pid=current_task_id,
        )

        yield_result(
            async_task,
            img_paths,
            current_progress,
            async_task.black_out_nsfw,
            False,
            do_not_show_finished_images=not show_intermediate_results
            or async_task.disable_intermediate_results,
        )

        return imgs, img_paths, current_progress

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task: ImageGenerationSeed):
        ip_adapter_manager = ip_adapter.IpaAdapterManagement()
        
        preparation_start_time = time.perf_counter()
        async_task.processing = True

        async_task.outpaint_selections = [o.lower() for o in async_task.outpaint_selections]
        
        base_model_additional_loras = []
        
        async_task.uov_method = async_task.uov_method.casefold()
        async_task.enhance_uov_method = async_task.enhance_uov_method.casefold()

        if fooocus_expansion in async_task.style_selections:
            use_expansion = True
            async_task.style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(async_task.style_selections) > 0

        

        current_progress = 0
      

        print(f"[Parameters] Adaptive CFG = {async_task.adaptive_cfg}")
        print(f"[Parameters] CLIP Skip = {async_task.clip_skip}")
        print(f"[Parameters] Sharpness = {async_task.sharpness}")
        print(f"[Parameters] ControlNet Softness = {async_task.controlnet_softness}")
        print(
            f"[Parameters] ADM Scale = "
            f"{async_task.adm_scaler_positive} : "
            f"{async_task.adm_scaler_negative} : "
            f"{async_task.adm_scaler_end}"
        )
        print(f"[Parameters] Seed = {async_task.seed}")

        apply_patch_settings(PatchSettings(**async_task), modules.patch.patch_settings)


        print(f"[Parameters] CFG = {async_task.cfg_scale}")

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = async_task.aspect_ratios_selection
        width, height = int(width), int(height)

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = async_task.inpaint_engine != "None"
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = (
            None,
            None,
            None,
            None,
        )

        goals = []
        tasks = []
        current_progress = 1

        if async_task.input_image_checkbox:
            (
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
            ) = apply_image_input(
                async_task,
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
            )

        # Load or unload CNs
        progressbar(async_task, current_progress, "Loading control models ...")
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])

        ip_adapter_manager.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter_manager.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        async_task.steps, switch, width, height = apply_overrides(
            async_task, async_task.steps, height, width
        )

        print(f"[Parameters] Sampler = {async_task.sampler_name} - {async_task.scheduler_name}")
        print(f"[Parameters] Steps = {async_task.steps} - {switch}")

        progressbar(async_task, current_progress, "Initializing ...")

        loras = async_task.loras
        if not skip_prompt_processing:
            tasks, use_expansion, loras, current_progress = process_prompt(
                async_task,
                async_task.prompt,
                async_task.negative_prompt,
                base_model_additional_loras,
                async_task.image_number,
                async_task.disable_seed_increment,
                use_expansion,
                use_style,
                use_synthetic_refiner,
                current_progress,
                advance_progress=True,
            )

        if len(goals) > 0:
            current_progress += 1
            progressbar(async_task, current_progress, "Image processing ...")

        if "vary" in goals:
            (
                async_task.uov_input_image,
                denoising_strength,
                initial_latent,
                width,
                height,
                current_progress,
            ) = apply_vary(
                async_task,
                async_task.uov_method,
                denoising_strength,
                async_task.uov_input_image,
                switch,
                current_progress,
            )

        if "upscale" in goals:
            (
                direct_return,
                async_task.uov_input_image,
                denoising_strength,
                initial_latent,
                tiled,
                width,
                height,
                current_progress,
            ) = apply_upscale(
                async_task,
                async_task.uov_input_image,
                async_task.uov_method,
                switch,
                current_progress,
                advance_progress=True,
            )
            if direct_return:
                d = [("Upscale (Fast)", "upscale_fast", "2x")]
                if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
                    progressbar(async_task, 100, "Checking for NSFW content ...")
                    async_task.uov_input_image = default_censor(
                        async_task.uov_input_image
                    )
                progressbar(async_task, 100, "Saving image to system ...")
                uov_input_image_path = log(
                    async_task.uov_input_image,
                    async_task.path_outputs,
                    d,
                    output_format=async_task.output_format,
                )
                yield_result(
                    async_task,
                    uov_input_image_path,
                    100,
                    async_task.black_out_nsfw,
                    False,
                    do_not_show_finished_images=True,
                )
                return

        if "inpaint" in goals:
            try:
                denoising_strength, initial_latent, width, height, current_progress = (
                    apply_inpaint(
                        async_task,
                        initial_latent,
                        inpaint_head_model_path,
                        inpaint_image,
                        inpaint_mask,
                        inpaint_parameterized,
                        async_task.inpaint_strength,
                        async_task.inpaint_respective_field,
                        switch,
                        async_task.inpaint_disable_initial_latent,
                        current_progress,
                        advance_progress=True,
                    )
                )
            except EarlyReturnException:
                return

        if async_task.cn_tasks:
            apply_control_nets(
                async_task,
                height,
                ip_adapter_face_path,
                ip_adapter_path,
                width,
                current_progress,
            )
            if async_task.debugging_cn_preprocessor:
                return

        if async_task.freeu_enabled:
            apply_freeu(async_task)

        # async_task.steps can have value of uov steps here when upscale has been applied
        steps, _, _, _ = apply_overrides(async_task, async_task.steps, height, width)

        images_to_enhance = []
        if "enhance" in goals:
            async_task.image_number = 1
            images_to_enhance += [async_task.enhance_input_image]
            height, width, _ = async_task.enhance_input_image.shape
            # input image already provided, processing is skipped
            steps = 0
            yield_result(
                async_task,
                async_task.enhance_input_image,
                current_progress,
                async_task.black_out_nsfw,
                False,
                async_task.disable_intermediate_results,
            )

        all_steps = steps * async_task.image_number

        if (async_task.enhance_checkbox and async_task.enhance_uov_method
        ):
            enhance_upscale_steps = async_task.performance_selection.steps()
            if "upscale" in async_task.enhance_uov_method:
                if "fast" in async_task.enhance_uov_method:
                    enhance_upscale_steps = 0
                else:
                    enhance_upscale_steps = async_task.performance_selection.steps_uov()
            enhance_upscale_steps, _, _, _ = apply_overrides(
                async_task, enhance_upscale_steps, height, width
            )
            enhance_upscale_steps_total = (
                async_task.image_number * enhance_upscale_steps
            )
            all_steps += enhance_upscale_steps_total

        if async_task.enhance_checkbox and len(async_task.enhance_ctrls) != 0:
            enhance_steps, _, _, _ = apply_overrides(
                async_task, async_task.original_steps, height, width
            )
            all_steps += (
                async_task.image_number * len(async_task.enhance_ctrls) * enhance_steps
            )

        all_steps = max(all_steps, 1)

        print(f"[Parameters] Denoising Strength = {denoising_strength}")

        if isinstance(initial_latent, dict) and "samples" in initial_latent:
            log_shape = initial_latent["samples"].shape
        else:
            log_shape = f"Image Space {(height, width)}"

        print(f"[Parameters] Initial Latent shape: {log_shape}")

        preparation_time = time.perf_counter() - preparation_start_time
        print(f"Preparation time: {preparation_time:.2f} seconds")

        final_scheduler_name = patch_samplers(async_task)
        print(f"Using {final_scheduler_name} scheduler.")

        async_task.yields.append(
            ["preview", (current_progress, "Moving model to GPU ...", None)]
        )

        processing_start_time = time.perf_counter()

        preparation_steps = current_progress
        total_count = async_task.image_number

        def callback(step, x0, x, total_steps, y):
            if step == 0:
                async_task.callback_steps = 0
            async_task.callback_steps += (100 - preparation_steps) / float(all_steps)
            async_task.yields.append(
                [
                    "preview",
                    (
                        int(current_progress + async_task.callback_steps),
                        f"Sampling step {step + 1}/{total_steps}, image {current_task_id + 1}/{total_count} ...",
                        y,
                    ),
                ]
            )

        show_intermediate_results = len(tasks) > 1 or async_task.should_enhance
        persist_image = (
            not async_task.should_enhance
            or not async_task.save_final_enhanced_image_only
        )

        for current_task_id, task in enumerate(tasks):
            progressbar(
                async_task,
                current_progress,
                f"Preparing task {current_task_id + 1}/{async_task.image_number} ...",
            )
            execution_start_time = time.perf_counter()

            try:
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
                    async_task.steps,
                    switch,
                    task["c"],
                    task["uc"],
                    task,
                    loras,
                    tiled,
                    use_expansion,
                    width,
                    height,
                    current_progress,
                    preparation_steps,
                    async_task.image_number,
                    show_intermediate_results,
                    persist_image,
                )

                current_progress = int(
                    preparation_steps
                    + (100 - preparation_steps)
                    / float(all_steps)
                    * async_task.steps
                    * (current_task_id + 1)
                )
                images_to_enhance += imgs

            except ldm_patched.modules.model_management.InterruptProcessingException:
                if async_task.last_stop == "skip":
                    print("User skipped")
                    async_task.last_stop = False
                    continue
                else:
                    print("User stopped")
                    break

            del task["c"], task["uc"]  # Save memory
            execution_time = time.perf_counter() - execution_start_time
            print(f"Generating and saving time: {execution_time:.2f} seconds")

        if not async_task.should_enchance:
            print(f"[Enhance] Skipping, preconditions aren't met")
            stop_processing(async_task, processing_start_time)
            return

        progressbar(async_task, current_progress, "Processing enhance ...")

        active_enhance_tabs = len(async_task.enhance_ctrls)
        should_process_enhance_uov = (
            async_task.enhance_uov_method != flags.disabled.casefold()
        )
        enhance_uov_before = False
        enhance_uov_after = False
        if should_process_enhance_uov:
            active_enhance_tabs += 1
            enhance_uov_before = (
                async_task.enhance_uov_processing_order == flags.enhancement_uov_before
            )
            enhance_uov_after = (
                async_task.enhance_uov_processing_order == flags.enhancement_uov_after
            )
        total_count = len(images_to_enhance) * active_enhance_tabs
        async_task.images_to_enhance_count = len(images_to_enhance)

        base_progress = current_progress
        current_task_id = -1
        done_steps_upscaling = 0
        done_steps_inpainting = 0
        enhance_steps, _, _, _ = apply_overrides(
            async_task, async_task.original_steps, height, width
        )
        exception_result = None
        for index, img in enumerate(images_to_enhance):
            async_task.enhance_stats[index] = 0
            enhancement_image_start_time = time.perf_counter()

            last_enhance_prompt = async_task.prompt
            last_enhance_negative_prompt = async_task.negative_prompt

            if enhance_uov_before:
                current_task_id += 1
                persist_image = (
                    not async_task.save_final_enhanced_image_only
                    or active_enhance_tabs == 0
                )
                (
                    current_task_id,
                    done_steps_inpainting,
                    done_steps_upscaling,
                    img,
                    exception_result,
                ) = enhance_upscale(
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
                    async_task.prompt,
                    async_task.negative_prompt,
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
                    persist_image,
                )
                async_task.enhance_stats[index] += 1

                if exception_result == "continue":
                    continue
                elif exception_result == "break":
                    break

            # inpaint for all other tabs
            for (
                enhance_mask_dino_prompt_text,
                enhance_prompt,
                enhance_negative_prompt,
                enhance_mask_model,
                enhance_mask_cloth_category,
                enhance_mask_sam_model,
                enhance_mask_text_threshold,
                enhance_mask_box_threshold,
                enhance_mask_sam_max_detections,
                enhance_inpaint_disable_initial_latent,
                enhance_inpaint_engine,
                enhance_inpaint_strength,
                enhance_inpaint_respective_field,
                enhance_inpaint_erode_or_dilate,
                enhance_mask_invert,
            ) in async_task.enhance_ctrls:
                current_task_id += 1
                current_progress = int(
                    base_progress
                    + (100 - preparation_steps)
                    / float(all_steps)
                    * (done_steps_upscaling + done_steps_inpainting)
                )
                progressbar(
                    async_task,
                    current_progress,
                    f"Preparing enhancement {current_task_id + 1}/{total_count} ...",
                )
                enhancement_task_start_time = time.perf_counter()
                is_last_enhance_for_image = (
                    current_task_id + 1
                ) % active_enhance_tabs == 0 and not enhance_uov_after
                persist_image = (
                    not async_task.save_final_enhanced_image_only
                    or is_last_enhance_for_image
                )

                extras = {}
                if enhance_mask_model == "sam":
                    print(f'[Enhance] Searching for "{enhance_mask_dino_prompt_text}"')
                elif enhance_mask_model == "u2net_cloth_seg":
                    extras["cloth_category"] = enhance_mask_cloth_category

                (
                    mask,
                    dino_detection_count,
                    sam_detection_count,
                    sam_detection_on_mask_count,
                ) = generate_mask_from_image(
                    img,
                    mask_model=enhance_mask_model,
                    extras=extras,
                    sam_options=SAMOptions(
                        dino_prompt=enhance_mask_dino_prompt_text,
                        dino_box_threshold=enhance_mask_box_threshold,
                        dino_text_threshold=enhance_mask_text_threshold,
                        dino_erode_or_dilate=async_task.dino_erode_or_dilate,
                        dino_debug=async_task.debugging_dino,
                        max_detections=enhance_mask_sam_max_detections,
                        model_type=enhance_mask_sam_model,
                    ),
                )
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]

                if int(enhance_inpaint_erode_or_dilate) != 0:
                    mask = erode_or_dilate(mask, enhance_inpaint_erode_or_dilate)

                if enhance_mask_invert:
                    mask = 255 - mask

                if async_task.debugging_enhance_masks_checkbox:
                    async_task.yields.append(
                        ["preview", (current_progress, "Loading ...", mask)]
                    )
                    yield_result(
                        async_task,
                        mask,
                        current_progress,
                        async_task.black_out_nsfw,
                        False,
                        async_task.disable_intermediate_results,
                    )
                    async_task.enhance_stats[index] += 1

                print(f"[Enhance] {dino_detection_count} boxes detected")
                print(f"[Enhance] {sam_detection_count} segments detected in boxes")
                print(
                    f"[Enhance] {sam_detection_on_mask_count} segments applied to mask"
                )

                if enhance_mask_model == "sam" and (
                    dino_detection_count == 0
                    or not async_task.debugging_dino
                    and sam_detection_on_mask_count == 0
                ):
                    print(
                        f'[Enhance] No "{enhance_mask_dino_prompt_text}" detected, skipping'
                    )
                    continue

                goals_enhance = ["inpaint"]

                try:
                    (
                        current_progress,
                        img,
                        enhance_prompt_processed,
                        enhance_negative_prompt_processed,
                    ) = process_enhance(
                        all_steps,
                        async_task,
                        callback,
                        controlnet_canny_path,
                        controlnet_cpds_path,
                        current_progress,
                        current_task_id,
                        denoising_strength,
                        enhance_inpaint_disable_initial_latent,
                        enhance_inpaint_engine,
                        enhance_inpaint_respective_field,
                        enhance_inpaint_strength,
                        enhance_prompt,
                        enhance_negative_prompt,
                        final_scheduler_name,
                        goals_enhance,
                        height,
                        img,
                        mask,
                        preparation_steps,
                        enhance_steps,
                        switch,
                        tiled,
                        total_count,
                        use_expansion,
                        use_style,
                        use_synthetic_refiner,
                        width,
                        persist_image=persist_image,
                    )
                    async_task.enhance_stats[index] += 1

                    if (
                        should_process_enhance_uov
                        and async_task.enhance_uov_processing_order
                        == flags.enhancement_uov_after
                        and async_task.enhance_uov_prompt_type
                        == flags.enhancement_uov_prompt_type_last_filled
                    ):
                        if enhance_prompt_processed != "":
                            last_enhance_prompt = enhance_prompt_processed
                        if enhance_negative_prompt_processed != "":
                            last_enhance_negative_prompt = (
                                enhance_negative_prompt_processed
                            )

                except (
                    ldm_patched.modules.model_management.InterruptProcessingException
                ):
                    if async_task.last_stop == "skip":
                        print("User skipped")
                        async_task.last_stop = False
                        continue
                    else:
                        print("User stopped")
                        exception_result = "break"
                        break
                finally:
                    done_steps_inpainting += enhance_steps

                enhancement_task_time = (
                    time.perf_counter() - enhancement_task_start_time
                )
                print(f"Enhancement time: {enhancement_task_time:.2f} seconds")

            if exception_result == "break":
                break

            if enhance_uov_after:
                current_task_id += 1
                # last step in enhance, always save
                persist_image = True
                (
                    current_task_id,
                    done_steps_inpainting,
                    done_steps_upscaling,
                    img,
                    exception_result,
                ) = enhance_upscale(
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
                    last_enhance_prompt,
                    last_enhance_negative_prompt,
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
                    persist_image,
                )
                async_task.enhance_stats[index] += 1

                if exception_result == "continue":
                    continue
                elif exception_result == "break":
                    break

            enhancement_image_time = time.perf_counter() - enhancement_image_start_time
            print(f"Enhancement image time: {enhancement_image_time:.2f} seconds")

        stop_processing(async_task, processing_start_time)
        return

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)

            try:
                handler(task)
                if task.generate_image_grid and len(task.results) > 2:
                    wall = build_image_wall(task.results)
                    task.results.append(wall)

                task.yields.append(["finish", task.results])
                pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.yields.append(["finish", task.results])
            finally:
                if pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[pid]
    pass


threading.Thread(target=worker, daemon=True).start()



### OLD CODE END ###


# NEW CODE START #

class ProcessTaskParams(BaseModel):
    all_steps: int
    async_task: ImageGenerationSeed = None,
    callback: callable = None,
    controlnet_canny_path: str = None,
    controlnet_cpds_path: str = None,
    current_task_id: int = None,
    denoising_strength: float = None,
    final_scheduler_name: str = None,
    goals: list = None,
    initial_latent: dict = None,
    steps: int = None,
    switch: int = None,
    positive_cond: dict = None,
    negative_cond: dict = None,
    task: dict = None,
    loras: bool = None,
    tiled: bool = None,
    use_expansion: bool = None,
    width: int = None,
    height: int = None,
    base_progress: int = None,
    preparation_steps: int = None,
    total_count: int = None,
    show_intermediate_results: bool = True,
    persist_image: bool = False



class ImageTaskProcessor:
    def __init__(self, async_tasks: list[ImageGenerationSeed], pipeline: pipeline):
        self.async_tasks = async_tasks
        self.pipeline = pipeline
        self.current_progress = 0
        
    def process_task(self, task: ImageGenerationSeed):
        if async_task.last_stop is not False:
            ldm_patched.modules.model_management.interrupt_current_processing()

        if async_task.cn_tasks:
            for cn_flag, cn_path in [
                (flags.cn_canny, controlnet_canny_path),
                (flags.cn_cpds, controlnet_cpds_path),
            ]:

                controlnet_task: ControlNetTask
                for controlnet_task in async_task.cn_tasks:
                    if controlnet_task.cn_type == cn_flag:
                        positive_cond, negative_cond = core.apply_controlnet(
                            positive_cond,
                            negative_cond,
                            pipeline.loaded_ControlNets[cn_path],
                            controlnet_task.cn_img,
                            controlnet_task.cn_weight,
                            0,
                            controlnet_task.cn_stop,
                        )

        imgs = pipeline.process_diffusion(
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            steps=steps,
            switch=switch,
            width=width,
            height=height,
            image_seed=task["task_seed"],
            callback=callback,
            sampler_name=async_task.sampler_name,
            scheduler_name=final_scheduler_name,
            latent=initial_latent,
            denoise=denoising_strength,
            tiled=tiled,
            cfg_scale=async_task.cfg_scale,
            refiner_swap_method=async_task.refiner_swap_method,
            disable_preview=async_task.disable_preview,
        )

        del positive_cond, negative_cond  # Save memory
        if inpaint_worker.current_task is not None:
            imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

        current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * steps)

        if config.default_black_out_nsfw or async_task.black_out_nsfw:
            progressbar(async_task, current_progress, "Checking for NSFW content ...")
            imgs = default_censor(imgs)

        progressbar(async_task, current_progress, f"Saving image {current_task_id + 1}/{total_count} to system ...")

        img_paths = save_and_log(
            async_task,
            height,
            imgs,
            task,
            use_expansion,
            width,
            loras,
            persist_image,
            fooocus_expansion=fooocus_expansion,
            pid=current_task_id,
        )

        yield_result(
            async_task,
            img_paths,
            current_progress,
            async_task.black_out_nsfw,
            False,
            do_not_show_finished_images=not show_intermediate_results
            or async_task.disable_intermediate_results,
        )

        return imgs, img_paths, current_progress





