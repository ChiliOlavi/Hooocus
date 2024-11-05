# OLD CODE START ###

import copy
import os
import random
import sys
import threading
from tkinter import Image
import traceback
from typing import List
from httpx import patch
import torch
import time

from extras.expansion import safe_str
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
    save_and_log,
    stop_processing,
)
from modules.inpaint_worker import InpaintWorker
from utils.model_file_config import BaseControlNetTask

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.config as config
import utils.flags as flags
from utils.logging_util import LoggingUtil
from utils.sdxl_prompt_expansion_utils import apply_arrays, apply_style, fooocus_expansion, get_random_style
from utils.flags import LORA_FILENAMES, DefaultControlNetTasks

from extras.inpaint_mask import generate_mask_from_image, SAMOptions
from extras.censor import default_censor
import extras.ip_adapter as ip_adapter

from modules.patch import PatchSettings, patch_all
from modules.util import apply_wildcards, erode_or_dilate, parse_lora_references_from_prompt, remove_empty_str, remove_performance_lora
from modules.private_logger import log
from modules.image_generation_utils import progressbar
from modules.default_pipeline import DefaultPipeline
from modules.core import apply_controlnet
import modules.patch

import ldm_patched.modules.model_management

patch_all()

GlobalConfig = config.LAUNCH_ARGS
GeneralCongig = GlobalConfig.GeneralArgs

logger = LoggingUtil().get_logger()

class EarlyReturnException(BaseException):
    pass

class ImageTaskProcessor:
    def __init__(self):
        self.initialize_processor()

    def initialize_processor(self):
        self.pid = os.getpid()
        self.pipeline = DefaultPipeline()
        self.goals = []
        self.results = []
        self.yields: list = []
        self.imgs = []
        self.use_prompt_expansion: bool = False
        self.current_progress: int = 1
        self.use_styles: bool = False
        self.patch_settings = PatchSettings()
        self.generation_task: config.ImageGenerationObject = None
        self.inpaint_worker: InpaintWorker = None
        self.generation_tasks: List[config.ImageGenerationObject] = []
        self.use_synthetic_refiner: bool = False
        self.base_model_additional_loras = []

    # Ok
    def yield_result(self, imgs: list, progressbar_index: int, do_not_show_finished_images=False):
        """Processes a list of images, optionally censors NSFW content, and updates the results."""
        imgs = self.censor_images_if_needed(imgs, progressbar_index)
        self.results.extend(imgs)
        if not do_not_show_finished_images:
            self.yields.append(["results", self.results])

    def censor_images_if_needed(self, imgs: list, progressbar_index: int) -> list:
        """Censors NSFW content in images if the configuration requires it."""
        if GlobalConfig.GeneralArgs.black_out_nsfw:
            progressbar(progressbar_index, "Checking for NSFW content ...")
            imgs = default_censor(imgs)
        return imgs

    def process_task(self, all_steps, callback, current_task_id, denoising_strength, final_scheduler_name,
                     initial_latent, steps, switch, positive_cond, negative_cond, task, loras, tiled,
                     use_expansion, width, height, base_progress, preparation_steps, total_count,
                     show_intermediate_results, persist_image=True):
        """Processes a single image generation task."""
        self.interrupt_if_needed()
        positive_cond, negative_cond = self.apply_controlnet_if_needed(positive_cond, negative_cond)
        imgs = self.generate_images(positive_cond, negative_cond, steps, switch, width, height, task, callback,
                                    final_scheduler_name, initial_latent, denoising_strength, tiled)
        imgs = self.post_process_images(imgs)
        current_progress = self.calculate_progress(base_progress, all_steps, steps, preparation_steps)
        self.yields.append(progressbar(current_progress, f"Saving image {current_task_id + 1}/{total_count} to system ..."))
        img_paths = self.save_images(imgs, task, use_expansion, width, height, loras, persist_image, current_task_id)
        self.yield_result(img_paths, current_progress)
        return imgs, img_paths, current_progress

    def interrupt_if_needed(self):
        """Interrupts the current processing if the last stop is not False."""
        if self.generation_task.last_stop is not False:
            ldm_patched.modules.model_management.interrupt_current_processing()

    def apply_controlnet_if_needed(self, positive_cond, negative_cond):
        """Applies controlnet tasks if they exist."""
        if self.generation_task.controlnet_tasks:
            for controlnet_task in self.generation_task.controlnet_tasks:
                if controlnet_task.name in [DefaultControlNetTasks.CPDS.name, DefaultControlNetTasks.PyraCanny.name]:
                    positive_cond, negative_cond = apply_controlnet(
                        positive_cond,
                        negative_cond,
                        self.pipeline.loaded_ControlNets[controlnet_task.name],
                        controlnet_task.img,
                        controlnet_task.weight,
                        0,
                        controlnet_task.stop,
                    )
        return positive_cond, negative_cond

    def generate_images(self, positive_cond, negative_cond, steps, switch, width, height, task, callback,
                        final_scheduler_name, initial_latent, denoising_strength, tiled):
        """Generates images using the pipeline."""
        return self.pipeline.process_diffusion(
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            steps=steps,
            switch=switch,
            width=width,
            height=height,
            image_seed=task["task_seed"],
            callback=callback,
            sampler_name=self.generation_task.sampler_name,
            scheduler_name=final_scheduler_name,
            latent=initial_latent,
            denoise=denoising_strength,
            tiled=tiled,
            cfg_scale=self.generation_task.cfg_scale,
            refiner_swap_method=self.generation_task.refiner_swap_method,
            disable_preview=self.generation_task.developer_options.disable_preview,
        )

    def post_process_images(self, imgs):
        """Post-processes images using the inpaint worker if available."""
        if self.inpaint_worker:
            imgs = [self.inpaint_worker.post_process(x) for x in imgs]
        return imgs

    def calculate_progress(self, base_progress, all_steps, steps, preparation_steps):
        """Calculates the current progress of the task."""
        return int(base_progress + (100 - preparation_steps) / float(all_steps) * steps)

    def save_images(self, imgs, task, use_expansion, width, height, loras, persist_image, current_task_id):
        """Saves and logs the generated images."""
        return save_and_log(
            self.generation_task,
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


    # DONE
    def process_prompt(self):
        """Processes the prompt and prepares tasks for image generation."""
        prompt, negative_prompt, use_expansion = self.prepare_prompts()
        loras = self.prepare_loras(prompt)
        self.pipeline.refresh_everything(
            refiner_model_name=self.generation_task.refiner_model,
            base_model_name=self.generation_task.base_model_name,
            loras=loras,
            base_model_additional_loras=self.base_model_additional_loras,
            use_synthetic_refiner=self.use_synthetic_refiner,
            vae_name=self.generation_task.vae_name
        )
        self.pipeline.set_clip_skip(self.generation_task.clip_skip)
        logger.info(f"Processing prompts ...")
        tasks = self.create_tasks(prompt, negative_prompt, use_expansion)
        return tasks, use_expansion, loras

    def prepare_prompts(self):
        """Prepares and returns the main and negative prompts."""
        prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default="")
        negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default="")
        prompt = prompts[0]
        negative_prompt = negative_prompts[0]
        use_expansion = self.use_prompt_expansion and prompt != ""
        return prompt, negative_prompt, use_expansion

    def prepare_loras(self, prompt):
        """Prepares and returns the loras for the task."""
        updated_lora_filenames = remove_performance_lora(LORA_FILENAMES, self.generation_task.performance_selection)
        loras, prompt = parse_lora_references_from_prompt(prompt, self.generation_task.loras, flags.max_lora_number, lora_filenames=updated_lora_filenames)
        loras += self.generation_task.performance_loras
        return loras

    def create_tasks(self, prompt, negative_prompt, use_expansion):
        """Creates and returns tasks for image generation."""
        tasks = []
        for i in range(self.generation_task.image_number):
            task_seed, task_rng = self.get_task_seed_and_rng(i)
            task_prompt, task_negative_prompt, task_extra_positive_prompts, task_extra_negative_prompts = self.get_task_prompts(
                prompt, negative_prompt, task_rng, i)
            positive_basic_workloads, negative_basic_workloads = self.get_basic_workloads(
                task_prompt, task_negative_prompt, task_extra_positive_prompts, task_extra_negative_prompts)
            task_styles = self.get_task_styles(task_prompt, positive_basic_workloads, task_rng)
            tasks.append(self.create_task_dict(task_seed, task_prompt, task_negative_prompt, positive_basic_workloads,
                                               negative_basic_workloads, task_styles, task_extra_positive_prompts,
                                               task_extra_negative_prompts))
        if use_expansion:
            self.expand_prompts(tasks)
        self.encode_prompts(tasks)
        return tasks

    def get_task_seed_and_rng(self, i):
        """Returns the task seed and random number generator."""
        if self.generation_task.developer_options.disable_seed_increment:
            task_seed = self.generation_task.seed
        else:
            task_seed = (self.generation_task.seed + i) % (GeneralCongig.max_seed + 1)
        task_rng = random.Random(task_seed)
        return task_seed, task_rng

    def get_task_prompts(self, prompt, negative_prompt, task_rng, i):
        """Returns the task prompts and extra prompts."""
        task_prompt = apply_wildcards(prompt, task_rng, i, self.generation_task.developer_options.read_wildcards_in_order)
        task_prompt = apply_arrays(task_prompt, i)
        task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, self.generation_task.developer_options.read_wildcards_in_order)
        extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, self.generation_task.developer_options.read_wildcards_in_order) for pmt in
                                  remove_empty_str([safe_str(p) for p in prompt.splitlines()][1:], default="")]
        extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, self.generation_task.developer_options.read_wildcards_in_order) for pmt in
                                  remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()][1:], default="")]
        return task_prompt, task_negative_prompt, extra_positive_prompts, extra_negative_prompts

    def get_basic_workloads(self, task_prompt, task_negative_prompt, task_extra_positive_prompts, task_extra_negative_prompts):
        """Returns the basic workloads for positive and negative prompts."""
        positive_basic_workloads = [task_prompt] + task_extra_positive_prompts
        negative_basic_workloads = [task_negative_prompt] + task_extra_negative_prompts
        return remove_empty_str(positive_basic_workloads, default=task_prompt), remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

    def get_task_styles(self, task_prompt, positive_basic_workloads, task_rng):
        """Returns the task styles."""
        task_styles = self.generation_task.styles.copy()
        if self.generation_task.use_styles:
            placeholder_replaced = False
            for j, s in enumerate(task_styles):
                if s == flags.random_style_name:
                    s = get_random_style(task_rng)
                    task_styles[j] = s
                p, n, style_has_placeholder = apply_style(s, positive=task_prompt)
                if style_has_placeholder:
                    placeholder_replaced = True
                positive_basic_workloads.extend(p)
            if not placeholder_replaced:
                positive_basic_workloads.insert(0, task_prompt)
        return task_styles

    def create_task_dict(self, task_seed, task_prompt, task_negative_prompt, positive_basic_workloads,
                         negative_basic_workloads, task_styles, task_extra_positive_prompts, task_extra_negative_prompts):
        """Creates and returns a task dictionary."""
        return dict(
            task_seed=task_seed,
            task_prompt=task_prompt,
            task_negative_prompt=task_negative_prompt,
            positive=positive_basic_workloads,
            negative=negative_basic_workloads,
            expansion="",
            c=None,
            uc=None,
            positive_top_k=len(positive_basic_workloads),
            negative_top_k=len(negative_basic_workloads),
            log_positive_prompt="\n".join([task_prompt] + task_extra_positive_prompts),
            log_negative_prompt="\n".join([task_negative_prompt] + task_extra_negative_prompts),
            styles=task_styles,
        )

    def expand_prompts(self, tasks):
        """Expands the prompts for each task."""
        for i, t in enumerate(tasks):
            logger.info(f"Expanding prompt {i + 1} ...")
            expansion = self.pipeline.final_expansion(t["task_prompt"], t["task_seed"])
            logger.info(f"[Prompt Expansion] {expansion}")
            t["expansion"] = expansion
            t["positive"] = copy.deepcopy(t["positive"]) + [expansion]

    def encode_prompts(self, tasks):
        """Encodes the prompts for each task."""
        for i, t in enumerate(tasks):
            logger.info(f"Encoding positive #{i + 1} ...")
            t["c"] = self.pipeline.clip_encode(texts=t["positive"], pool_top_k=t["positive_top_k"])
        for i, t in enumerate(tasks):
            if abs(float(self.generation_task.cfg_scale) - 1.0) < 1e-4:
                t["uc"] = self.pipeline.clone_cond(t["c"])
            else:
                logger.info(f"Encoding negative #{i + 1} ...")
                t["uc"] = self.pipeline.clip_encode(texts=t["negative"], pool_top_k=t["negative_top_k"])

    # TODO? This one could be async, using its own pid etc to patch
    def process_all_tasks(self):
        """Processes all tasks in the generation queue."""
        time.sleep(0.01)
        while self.generation_tasks:
            self.process_single_task()

    def process_single_task(self):
        """Processes a single task from the generation queue."""
        task: config.ImageGenerationObject = self.generation_tasks.pop(0)
        self.generation_task = task
        new_patch_settings = apply_patch_settings(self.patch_settings, task.patch_settings)
        try:
            self.handler(task)
            self.generate_image_wall_if_needed(task)
            task.yields.append(["finish", task.results])
            self.pipeline.prepare_text_encoder(async_call=True)
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            traceback.print_exc()
            task.yields.append(["finish", task.results])
        finally:
            self.cleanup_after_task()

    def generate_image_wall_if_needed(self, task):
        """Generates an image wall if the task requires it."""
        if task.developer_options.generate_grid and len(self.results) > 2:
            wall = build_image_wall(task.results)
            task.results.append(wall)

    def cleanup_after_task(self):
        """Cleans up after processing a task."""
        if self.pid in modules.patch.patch_settings:
            del modules.patch.patch_settings[self.pid]
    pass

    threading.Thread(target=worker, daemon=True).start()
