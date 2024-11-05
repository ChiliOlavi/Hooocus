


example_task_from_ui = {
    "args": [
        False, # Generate image grid
        "A car without headlights a banana man happy", # Prompt
        "", # Negative prompt
        [ # Style selections
            "Fooocus V2",
            "Fooocus Enhance",
            "Fooocus Sharp",
            "SAI Neonpunk"
        ],
        "Quality", # Performance selection
        "1152\u00d7896 <span style=\'color: grey;\'> \u2223 9:7</span>", # Aspect ratios selection
        2, # Image number
        "png", # Output format
        "4524879730836419225", # Seed
        False, # Read wildcards in order
        15.928, # Sharpness
        4, # CFG scale
        "juggernautXL_v8Rundiffusion.safetensors", # Base model name
        "None", # Refiner model name
        
        0.5, # loras ((bool, str, float) for default_max_lora_number)
        True,
        "sd_xl_offset_example-lora_1.0.safetensors",
        0.1,
        True,
        "None",
        1,
        True,
        "None",
        1,
        True,
        "None",
        1,
        True,
        "None", 
        1, # Loras end
        False, # Input image checkbox
        "uov", # Current tab
        "Disabled", # UOV method
        None, # UOV input image
        [], # Outpaint selections
        None, # Inpaint input image
        "", # Inpaint additional prompt
        None, # Inpaint mask image upload
        False, # Disable preview
        False, # Disable intermediate results
        False, # Disable seed increment
        False, # Black out NSFW
        1.5, # ADM scaler positive
        0.8, # ADM scaler negative
        0.3, # ADM scaler end
        7, # Adaptive CFG
        2, # Clip skip
        "dpmpp_2m_sde_gpu", # Sampler name
        "karras", # Scheduler name
        "Default (model)", # VAE name
        -1, # Overwrite step
        -1, # Overwrite switch
        -1, # Overwrite width
        -1, # Overwrite height
        -1, # Overwrite vary strength
        -1, # Overwrite upscale strength
        False, # Mixing image prompt and vary upscale
        False, # Mixing image prompt and inpaint
        False, # Debugging CN preprocessor
        False, # Skipping CN preprocessor
        64, # Canny low threshold
        128, # Canny high threshold
        "joint", # Refiner swap method
        0.25, # Controlnet softness
        False, # Freeu enabled
        1.01, # Freeu b1
        1.02, # Freeu b2
        0.99, # Freeu s1
        0.95, # Freeu s2
        False, # Debugging inpaint preprocessor
        False, # Inpaint disable initial latent
        "v2.6", # Inpaint engine
        1, # Inpaint strength
        0.618, # Inpaint respective field
        False, # Inpaint advanced masking checkbox
        False, # Invert mask checkbox
        0, # Inpaint erode or dilate
        False, # Save final enhanced image only
        False, # Save metadata to images
        "fooocus", # Metadata scheme
        None, # CN tasks start
        0.5,
        0.6,
        "ImagePrompt",
        None,
        0.5,
        0.6,
        "ImagePrompt",
        None,
        0.5,
        0.6,
        "ImagePrompt",
        None,
        0.5,
        0.6,
        "ImagePrompt", # CN tasks end
        False,
        0,
        False,
        None,
        False,
        "Disabled",
        "Before First Enhancement",
        "Original Prompts",
        False,
        "",
        "",
        "",
        "sam",
        "full",
        "vit_b",
        0.25,
        0.3,
        0,
        False,
        "v2.6",
        1,
        0.618,
        0,
        False,
        False,
        "",
        "",
        "",
        "sam",
        "full",
        "vit_b",
        0.25,
        0.3,
        0,
        False,
        "v2.6",
        1,
        0.618,
        0,
        False,
        False,
        "",
        "",
        "",
        "sam",
        "full",
        "vit_b",
        0.25,
        0.3,
        0,
        False,
        "v2.6",
        1,
        0.618,
        0,
        False
    ],
    "yields": [],
    "results": [],
    "last_stop": False,
    "processing": False,
    "performance_loras": [],
    "generate_image_grid": False,
    "prompt": "A car without headlights a banana man happy",
    "negative_prompt": "",
    "style_selections": [
        "Fooocus V2",
        "Fooocus Enhance",
        "Fooocus Sharp",
        "SAI Neonpunk"
    ],
    "performance_selection": "<Performance.QUALITY: 'Quality'>",
    "steps": 60,
    "original_steps": 60,
    "aspect_ratios_selection": "1152×896 <span style='color: grey;'> ∣ 9:7</span>",
    "image_number": 2,
    "output_format": "png",
    "seed": 4524879730836419225,
    "read_wildcards_in_order": False,
    "sharpness": 15.928,
    "cfg_scale": 4,
    "base_model_name": "juggernautXL_v8Rundiffusion.safetensors",
    "refiner_model_name": "None",
    "refiner_switch": 0.5,
    "loras": [("sd_xl_offset_example-lora_1.0.safetensors",
        0.1)
    ],
    "input_image_checkbox": False,
    "current_tab": "uov",
    "uov_method": "Disabled",
    "uov_input_image": None,
    "outpaint_selections": [],
    "inpaint_input_image": None,
    "inpaint_additional_prompt": "",
    "inpaint_mask_image_upload": None,
    "disable_preview": False,
    "disable_intermediate_results": False,
    "disable_seed_increment": False,
    "black_out_nsfw": False,
    "adm_scaler_positive": 1.5,
    "adm_scaler_negative": 0.8,
    "adm_scaler_end": 0.3,
    "adaptive_cfg": 7,
    "clip_skip": 2,
    "sampler_name": "dpmpp_2m_sde_gpu",
    "scheduler_name": "karras",
    "vae_name": "Default (model)",
    "overwrite_step": -1,
    "overwrite_switch": -1,
    "overwrite_width": -1,
    "overwrite_height": -1,
    "overwrite_vary_strength": -1,
    "overwrite_upscale_strength": -1,
    "mixing_image_prompt_and_vary_upscale": False,
    "mixing_image_prompt_and_inpaint": False,
    "debugging_cn_preprocessor": False,
    "skipping_cn_preprocessor": False,
    "canny_low_threshold": 64,
    "canny_high_threshold": 128,
    "refiner_swap_method": "joint",
    "controlnet_softness": 0.25,
    "freeu_enabled": False,
    "freeu_b1": 1.01,
    "freeu_b2": 1.02,
    "freeu_s1": 0.99,
    "freeu_s2": 0.95,
    "debugging_inpaint_preprocessor": False,
    "inpaint_disable_initial_latent": False,
    "inpaint_engine": "v2.6",
    "inpaint_strength": 1,
    "inpaint_respective_field": 0.618,
    "inpaint_advanced_masking_checkbox": False,
    "invert_mask_checkbox": False,
    "inpaint_erode_or_dilate": 0,
    "save_final_enhanced_image_only": False,
    "save_metadata_to_images": False,
    "metadata_scheme": "<MetadataScheme.FOOOCUS: 'fooocus'>",
    "cn_tasks": {
        "ImagePrompt": [],
        "PyraCanny": [],
        "CPDS": [],
        "FaceSwap": []
    },
    "debugging_dino": False,
    "dino_erode_or_dilate": 0,
    "debugging_enhance_masks_checkbox": False,
    "enhance_input_image": None,
    "enhance_checkbox": False,
    "enhance_uov_method": "Disabled",
    "enhance_uov_processing_order": "Before First Enhancement",
    "enhance_uov_prompt_type": "Original Prompts",
    "enhance_ctrls": [],
    "should_enhance": False,
    "images_to_enhance_count": 0,
    "enhance_stats": {}
}