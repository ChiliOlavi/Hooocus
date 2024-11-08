import modules.core as core
import os
import sys

from h3_utils import config
from h3_utils.path_configs import FolderPathsConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import modules.patch
import h3_utils.config
import h3_utils.flags
import ldm_patched.modules.model_management
import ldm_patched.modules.latent_formats
import modules.inpaint_worker
import extras.vae_interpose as vae_interpose
from extras.expansion import FooocusExpansion

from ldm_patched.modules.model_base import SDXL, SDXLRefiner
from modules.sample_hijack import clip_separate
from modules.util import get_file_from_folder_list, get_enabled_loras
from h3_utils.config import LAUNCH_ARGS, DefaultConfigImageGen
from h3_utils.logging_util import LoggingUtil

from unavoided_global_hell.unavoided_global_vars import patch_settings_GLOBAL_CAUTION

logger = LoggingUtil().get_logger()
logger.name = 'default_pipeline'

class DefaultPipeline:

    model_base = core.StableDiffusionModel()
    model_refiner = core.StableDiffusionModel()

    final_expansion = None

    final_model = None
    final_unet = None
    final_clip = None
    final_vae = None
    final_refiner_unet = None
    final_refiner_vae = None

    target_model = None
    target_unet = None
    target_vae = None
    target_refiner_unet = None
    target_refiner_vae = None
    target_clip = None

    loaded_ControlNets = {}

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_controlnets (self, model_paths):
        logger.debug(f'Refreshing ControlNets: {model_paths}')
        cache = {}
        for p in model_paths:
            if p is not None:
                if p in self.loaded_ControlNets:
                    cache[p] = self.loaded_ControlNets[p]
                else:
                    cache[p] = core.load_controlnet(p)
        self.loaded_ControlNets = cache
        return

    def assert_model_integrity (self):
        error_message = None

        if not isinstance(self.model_base.unet_with_lora.model, SDXL):
            error_message = 'You have selected base model other than SDXL. This is not supported yet.'

        if error_message is not None:
            raise NotImplementedError(error_message)
        return True

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_base_model (self, name, vae_name=None):
        filename = get_file_from_folder_list(name, FolderPathsConfig.path_checkpoints)

        vae_filename = None
        if vae_name is not None and vae_name != DefaultConfigImageGen.vae_name:
            vae_filename = get_file_from_folder_list(vae_name, FolderPathsConfig.path_vae.value)

        if self.model_base.filename == filename and self.model_base.vae_filename == vae_filename:
            return

        self.model_base = core.load_model(filename, vae_filename)
        print(f'Base model loaded: {self.model_base.filename}')
        print(f'VAE loaded: {self.model_base.vae_filename}')
        return

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_refiner_model (self, name):
        if not name:
            print(f'Refiner unloaded.')
            return
        
        filename = get_file_from_folder_list(name, FolderPathsConfig.path_checkpoints)
        
        if self.model_refiner.filename == filename:
            return
        
        self.model_refiner = None
        self.model_refiner = core.load_model(filename)
        print(f'Refiner model loaded: {self.model_refiner.filename}')
        
        if isinstance(self.model_refiner.unet.model, SDXL):
            self.model_refiner.clip = None
            self.model_refiner.vae = None
        elif isinstance(self.model_refiner.unet.model, SDXLRefiner):
            self.model_refiner.clip = None
            self.model_refiner.vae = None
        else:
            self.model_refiner.clip = None
        return

    @torch.no_grad()
    @torch.inference_mode()
    def synthesize_refiner_model (self):

        print('Synthetic Refiner Activated')
        self.model_refiner = core.StableDiffusionModel(
            unet=self.model_base.unet,
            vae=self.model_base.vae,
            clip=self.model_base.clip,
            clip_vision=self.model_base.clip_vision,
            filename=self.model_base.filename
        )
        self.model_refiner.vae = None
        self.model_refiner.clip = None
        self.model_refiner.clip_vision = None

        return

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras (self, loras, base_model_additional_loras=None):

        if not isinstance(base_model_additional_loras, list):
            base_model_additional_loras = []

        self.model_base.refresh_loras(loras + base_model_additional_loras)
        self.model_refiner.refresh_loras(loras)

        return

    @torch.no_grad()
    @torch.inference_mode()
    def clip_encode_single (self, clip, text, verbose=False):
        cached = clip.fcs_cond_cache.get(text, None)
        if cached is not None:
            if verbose:
                print(f'[CLIP Cached] {text}')
            return cached
        tokens = clip.tokenize(text)
        result = clip.encode_from_tokens(tokens, return_pooled=True)
        clip.fcs_cond_cache[text] = result
        if verbose:
            print(f'[CLIP Encoded] {text}')
        return result

    @torch.no_grad()
    @torch.inference_mode()
    def clone_cond (self, conds):
        results = []

        for c, p in conds:
            p = p["pooled_output"]

            if isinstance(c, torch.Tensor):
                c = c.clone()

            if isinstance(p, torch.Tensor):
                p = p.clone()

            results.append([c, {"pooled_output": p}])

        return results

    @torch.no_grad()
    @torch.inference_mode()
    def clip_encode (self, texts, pool_top_k=1):

        if self.final_clip is None:
            return None
        if not isinstance(texts, list):
            return None
        if len(texts) == 0:
            return None

        cond_list = []
        pooled_acc = 0

        for i, text in enumerate(texts):
            cond, pooled = self.clip_encode_single(self.final_clip, text)
            cond_list.append(cond)
            if i < pool_top_k:
                pooled_acc += pooled

        return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]

    @torch.no_grad()
    @torch.inference_mode()
    def set_clip_skip (self, clip_skip: int):

        if self.final_clip is None:
            return

        self.final_clip.clip_layer(-abs(clip_skip))
        return

    @torch.no_grad()
    @torch.inference_mode()
    def clear_all_caches (self):
        self.final_clip.fcs_cond_cache = {}

    @torch.no_grad()
    @torch.inference_mode()
    def prepare_text_encoder (self, async_call=True):
        if async_call:
            # TODO: make sure that this is always called in an async way so that users cannot feel it.
            pass
        self.assert_model_integrity()
        ldm_patched.modules.model_management.load_models_gpu([self.final_clip.patcher, self.final_expansion.patcher])
        return

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_everything(
        self,
        refiner_model_name,
        base_model_name,
        loras,
        base_model_additional_loras=None,
        use_synthetic_refiner=False,
        vae_name=None):

        self.final_unet = None
        self.final_clip = None
        self.final_vae = None
        self.final_refiner_unet = None
        self.final_refiner_vae = None

        if use_synthetic_refiner and not refiner_model_name:
            print('Synthetic Refiner Activated')
            self.refresh_base_model(base_model_name, vae_name)
            self.synthesize_refiner_model()
        else:
            self.refresh_refiner_model(refiner_model_name)
            self.refresh_base_model(base_model_name, vae_name)

        self.refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
        self.assert_model_integrity()

        self.final_unet = self.model_base.unet_with_lora
        self.final_clip = self.model_base.clip_with_lora
        self.final_vae = self.model_base.vae

        self.final_refiner_unet = self.model_refiner.unet_with_lora
        self.final_refiner_vae = self.model_refiner.vae

        if self.final_expansion is None:
            self.final_expansion = FooocusExpansion()

        self.prepare_text_encoder(async_call=True)
        self.clear_all_caches()
        return

    @torch.no_grad()
    @torch.inference_mode()
    def vae_parse (self, latent):
        if self.final_refiner_vae is None:
            return latent

        result = vae_interpose.parse(latent["samples"])
        return {'samples': result}

    @torch.no_grad()
    @torch.inference_mode()
    def calculate_sigmas_all (self, sampler, model, scheduler, steps):
        from ldm_patched.modules.samplers import calculate_sigmas_scheduler

        discard_penultimate_sigma = False
        if sampler in ['dpm_2', 'dpm_2_ancestral']:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas_scheduler(model, scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    @torch.no_grad()
    @torch.inference_mode()
    def calculate_sigmas (self, sampler, model, scheduler, steps, denoise):
        if denoise is None or denoise > 0.9999 or denoise < 0.000001:
            sigmas = self.calculate_sigmas_all(sampler, model, scheduler, steps)
        else:
            new_steps = int(steps / denoise)
            sigmas = self.calculate_sigmas_all(sampler, model, scheduler, new_steps)
            sigmas = sigmas[-(steps + 1):]
        return sigmas

    @torch.no_grad()
    @torch.inference_mode()
    def get_candidate_vae (self, steps, switch, denoise=1.0):
        if self.final_refiner_vae is not None and self.final_refiner_unet is not None:
            if denoise > 0.9:
                return self.final_vae, self.final_refiner_vae
            else:
                if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                    return self.final_vae, None
                else:
                    return self.final_refiner_vae, None

        return self.final_vae, self.final_refiner_vae

    @torch.no_grad()
    @torch.inference_mode()
    def process_diffusion(
        self,
        positive_cond,
        negative_cond,
        steps: int,
        switch: int,
        width: int,
        height: int,
        image_seed: int,
        callback: callable,
        sampler_name: str,
        scheduler_name: str,
        latent: bool | None = None,
        denoise: float = 1.0,
        tiled: bool = False,
        cfg_scale:float = 7.0,
        refiner_swap_method: str = "joint",
        inpaintworker: modules.inpaint_worker.InpaintWorker = None,
        disable_preview=False):

        self.target_model = self.final_model
        self.target_unet = self.final_unet
        self.target_vae = self.final_vae
        self.target_refiner_unet = self.final_refiner_unet
        self.target_refiner_vae = self.final_refiner_vae
        self.target_clip = self.final_clip


        if self.final_refiner_vae is not None and self.final_refiner_unet is not None:
            # Refiner Use Different VAE (then it is SD15)
            if denoise > 0.9:
                refiner_swap_method = 'vae'
            else:
                refiner_swap_method = "joint"
                if (
                    denoise > (float(steps - switch) / float(steps)) ** 0.834):  # karras 0.834
                    self.target_unet = self.final_unet
                    self.target_vae = self.final_vae
                    self.target_refiner_unet = None
                    self.target_refiner_vae = None

                    print(f"[Sampler] only use Base because of partial denoise.")
                else:
                    positive_cond = clip_separate(
                        positive_cond,
                        target_model=self.final_refiner_unet.model,
                        target_clip=self.final_clip)

                    negative_cond = clip_separate(
                        negative_cond,
                        target_model=self.final_refiner_unet.model,
                        target_clip=self.final_clip)

                    self.target_unet = self.final_refiner_unet
                    self.target_vae = self.final_refiner_vae
                    self.target_refiner_unet = None
                    self.target_refiner_vae  = None
            
                    print(f"[Sampler] only use Refiner because of partial denoise.")

        print(f'[Sampler] refiner_swap_method = {refiner_swap_method}')

        if latent is None:
            initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
        else:
            initial_latent = latent

        minmax_sigmas = self.calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=self.final_unet.model, steps=steps, denoise=denoise)
        sigma_min, sigma_max = minmax_sigmas[minmax_sigmas > 0].min(), minmax_sigmas.max()
        sigma_min = float(sigma_min.cpu().numpy())
        sigma_max = float(sigma_max.cpu().numpy())
        print(f'[Sampler] sigma_min = {sigma_min}, sigma_max = {sigma_max}')

        modules.patch.BrownianTreeNoiseSamplerPatched.global_init(
            initial_latent['samples'].to(ldm_patched.modules.model_management.get_torch_device()),
            sigma_min, sigma_max, seed=image_seed, cpu=False)

        decoded_latent = None

        if refiner_swap_method == 'joint':
            sampled_latent = core.ksampler(
                model=self.target_unet,
                refiner=self.target_refiner_unet,
                positive=positive_cond,
                negative=negative_cond,
                latent=initial_latent,
                steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                refiner_switch=switch,
                previewer_start=0,
                previewer_end=steps,
                disable_preview=disable_preview
            )
            decoded_latent = core.decode_vae(vae=self.target_vae, latent_image=sampled_latent, tiled=tiled)

        if refiner_swap_method == 'separate':
            sampled_latent = core.ksampler(
                model=self.target_unet,
                positive=positive_cond,
                negative=negative_cond,
                latent=initial_latent,
                steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=False,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=0,
                previewer_end=steps,
                disable_preview=disable_preview
            )
            print('Refiner swapped by changing ksampler. Noise preserved.')

            self.target_model = self.target_refiner_unet
            if not self.target_model:
                self.target_model = self.target_unet
                print('Use base model to refine itself - this may because of developer mode.')

            sampled_latent = core.ksampler(
                model=self.target_model,
                positive=clip_separate(positive_cond, target_model=self.target_model.model, target_clip=self.target_clip),
                negative=clip_separate(negative_cond, target_model=self.target_model.model, target_clip=self.target_clip),
                latent=sampled_latent,
                steps=steps, start_step=switch, last_step=steps, disable_noise=True, force_full_denoise=True,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=switch,
                previewer_end=steps,
                disable_preview=disable_preview
            )

            self.target_model = self.target_refiner_vae
            if not self.target_model:
                self.target_model = self.target_vae
            decoded_latent = core.decode_vae(vae=self.target_model, latent_image=sampled_latent, tiled=tiled)

        if refiner_swap_method == 'vae':
            # TODO PATH
            # GLOBAL VAR USAGE
            patch_settings_GLOBAL_CAUTION[os.getpid()].eps_record = 'vae'

            if inpaintworker and inpaintworker.current_task is not None:
                inpaintworker.unswap()

            sampled_latent = core.ksampler(
                model=self.target_unet,
                positive=positive_cond,
                negative=negative_cond,
                latent=initial_latent,
                steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=True,
                seed=image_seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=0,
                previewer_end=steps,
                disable_preview=disable_preview
            )
            print('Fooocus VAE-based swap.')

            self.target_model = self.target_refiner_unet
            if self.target_model is None:
                self.target_model = self.target_unet
                print('Use base model to refine itself - this may because of developer mode.')

            sampled_latent = self.vae_parse(sampled_latent)

            k_sigmas = 1.4
            sigmas = self.calculate_sigmas(sampler=sampler_name,
                                    scheduler=scheduler_name,
                                    model=self.target_model.model,
                                    steps=steps,
                                    denoise=denoise)[switch:] * k_sigmas
            len_sigmas = len(sigmas) - 1

            # GLOBAL VAR USAGE
            noise_mean = torch.mean(patch_settings_GLOBAL_CAUTION[os.getpid()].eps_record, dim=1, keepdim=True)

            if inpaintworker and inpaintworker.current_task is not None:
                inpaintworker.swap()

            sampled_latent = core.ksampler(
                model=self.target_model,
                positive=clip_separate(positive_cond, target_model=self.target_model.model, target_clip=self.target_clip),
                negative=clip_separate(negative_cond, target_model=self.target_model.model, target_clip=self.target_clip),
                latent=sampled_latent,
                steps=len_sigmas, start_step=0, last_step=len_sigmas, disable_noise=False, force_full_denoise=True,
                seed=image_seed+1,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=switch,
                previewer_end=steps,
                sigmas=sigmas,
                noise_mean=noise_mean,
                disable_preview=disable_preview
            )

            self.target_model = self.target_refiner_vae
            if self.target_model is None:
                self.target_model = self.target_vae
            decoded_latent = core.decode_vae(vae=self.target_model, latent_image=sampled_latent, tiled=tiled)

        images = core.pytorch_to_numpy(decoded_latent)
        
        # GLOBAL VAR USAGE
        patch_settings_GLOBAL_CAUTION[os.getpid()].eps_record = None
        return images

