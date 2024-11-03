import sys, os

from utils import config, flags

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.consts import HOOOCUS_VERSION, METADATA_SCHEME
from utils.hookus_utils import ImageGenerationSeed
from modules import patch, meta_parser
from modules.private_logger import log

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