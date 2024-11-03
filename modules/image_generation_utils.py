import sys, os

from utils import config, flags

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.consts import HOOOCUS_VERSION, METADATA_SCHEME
from utils.hookus_utils import ImageGenerationSeed
from modules import patch, meta_parser
from modules.private_logger import log

import numpy as np

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



def apply_image_input(
    async_task: ImageGenerationSeed,
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
            async_task.should_upscale_or_vary
            or (
                async_task.should_use_imageprompt   
                and async_task.mixing_image_prompt_and_vary_upscale
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
        async_task.should_inpaint or (async_task.should_use_imageprompt and async_task.mixing_image_prompt_and_inpaint)) and async_task.inpaint_input_image is not None:
        inpaint_image = async_task.inpaint_input_image["image"]
        inpaint_mask = async_task.inpaint_input_image["mask"][:, :, 0]

        if async_task.inpaint_advanced_masking_checkbox:
            if isinstance(async_task.inpaint_mask_image_upload, dict):
                if (
                    isinstance(
                        async_task.inpaint_mask_image_upload["image"], np.ndarray
                    )
                    and isinstance(
                        async_task.inpaint_mask_image_upload["mask"], np.ndarray
                    )
                    and async_task.inpaint_mask_image_upload["image"].ndim == 3
                ):
                    async_task.inpaint_mask_image_upload = np.maximum(
                        async_task.inpaint_mask_image_upload["image"],
                        async_task.inpaint_mask_image_upload["mask"],
                    )
            if (
                isinstance(async_task.inpaint_mask_image_upload, np.ndarray)
                and async_task.inpaint_mask_image_upload.ndim == 3
            ):
                H, W, C = inpaint_image.shape
                async_task.inpaint_mask_image_upload = resample_image(
                    async_task.inpaint_mask_image_upload, width=W, height=H
                )
                async_task.inpaint_mask_image_upload = np.mean(
                    async_task.inpaint_mask_image_upload, axis=2
                )
                async_task.inpaint_mask_image_upload = (
                    async_task.inpaint_mask_image_upload > 127
                ).astype(np.uint8) * 255
                inpaint_mask = np.maximum(
                    inpaint_mask, async_task.inpaint_mask_image_upload
                )

        if int(async_task.inpaint_erode_or_dilate) != 0:
            inpaint_mask = erode_or_dilate(
                inpaint_mask, async_task.inpaint_erode_or_dilate
            )

        if async_task.invert_mask_checkbox:
            inpaint_mask = 255 - inpaint_mask

        inpaint_image = ensure_three_channels(inpaint_image)
        if (
            isinstance(inpaint_image, np.ndarray)
            and isinstance(inpaint_mask, np.ndarray)
            and (
                np.any(inpaint_mask > 127)
                or len(async_task.outpaint_selections) > 0
            )
        ):
            progressbar(async_task, 1, "Downloading upscale models ...")
            config.downloading_upscale_model()
            if inpaint_parameterized:
                progressbar(async_task, 1, "Downloading inpainter ...")
                inpaint_head_model_path, inpaint_patch_model_path = (
                    config.downloading_inpaint_models(
                        async_task.inpaint_engine
                    )
                )
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                print(
                    f"[Inpaint] Current inpaint model is {inpaint_patch_model_path}"
                )
                if async_task.refiner_model_name == "None":
                    use_synthetic_refiner = True
                    async_task.refiner_switch = 0.8
            else:
                inpaint_head_model_path, inpaint_patch_model_path = None, None
                print(f"[Inpaint] Parameterized inpaint is disabled.")
            if async_task.inpaint_additional_prompt != "":
                if async_task.prompt == "":
                    async_task.prompt = async_task.inpaint_additional_prompt
                else:
                    async_task.prompt = (
                        async_task.inpaint_additional_prompt
                        + "\n"
                        + async_task.prompt
                    )
            goals.append("inpaint")
    if (
        async_task.current_tab == "ip"
        or async_task.mixing_image_prompt_and_vary_upscale
        or async_task.mixing_image_prompt_and_inpaint
    ):
        goals.append("cn")
        progressbar(async_task, 1, "Downloading control models ...")
        if len(async_task.cn_tasks[flags.cn_canny]) > 0:
            controlnet_canny_path = config.downloading_controlnet_canny()
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