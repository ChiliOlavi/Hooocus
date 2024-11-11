def match_lora(lora, to_load):
    """
    Match and process different types of LoRA (Low-Rank Adaptation) weights.
    
    Args:
        lora (dict): Dictionary containing the LoRA weights and parameters
        to_load (dict): Mapping of keys to their corresponding load destinations
    
    Returns:
        tuple: (patch_dict, remaining_dict)
            - patch_dict: Dictionary of processed LoRA weights
            - remaining_dict: Dictionary of unprocessed weights
    """
    patch_dict = {}
    loaded_keys = set()

    def add_to_loaded(keys):
        """Helper function to add keys to loaded_keys set"""
        for key in keys:
            loaded_keys.add(key)

    for x in to_load:
        real_load_key = to_load[x]
        
        # Handle direct matches
        if real_load_key in lora:
            patch_dict[real_load_key] = ('fooocus', lora[real_load_key])
            loaded_keys.add(real_load_key)
            continue

        # Get alpha value if present
        alpha_name = f"{x}.alpha"
        alpha = lora[alpha_name].item() if alpha_name in lora.keys() else None
        if alpha is not None:
            loaded_keys.add(alpha_name)

        # Handle standard LoRA weights
        lora_patterns = {
            'regular': (f"{x}.lora_up.weight", f"{x}.lora_down.weight", f"{x}.lora_mid.weight"),
            'diffusers': (f"{x}_lora.up.weight", f"{x}_lora.down.weight", None),
            'transformers': (f"{x}.lora_linear_layer.up.weight", f"{x}.lora_linear_layer.down.weight", None)
        }

        for pattern_type, (up_name, down_name, mid_name) in lora_patterns.items():
            if up_name in lora.keys():
                mid = lora[mid_name] if mid_name and mid_name in lora.keys() else None
                patch_dict[to_load[x]] = ("lora", (lora[up_name], lora[down_name], alpha, mid))
                add_to_loaded([up_name, down_name])
                if mid:
                    loaded_keys.add(mid_name)
                break

        # Handle LoHA (LoRA with Hadamard Product)
        hada_keys = {
            'w1_a': f"{x}.hada_w1_a",
            'w1_b': f"{x}.hada_w1_b",
            'w2_a': f"{x}.hada_w2_a",
            'w2_b': f"{x}.hada_w2_b",
            't1': f"{x}.hada_t1",
            't2': f"{x}.hada_t2"
        }

        if hada_keys['w1_a'] in lora.keys():
            hada_t1 = lora[hada_keys['t1']] if hada_keys['t1'] in lora.keys() else None
            hada_t2 = lora[hada_keys['t2']] if hada_keys['t2'] in lora.keys() else None
            
            if hada_t1 is not None:
                add_to_loaded([hada_keys['t1'], hada_keys['t2']])

            patch_dict[to_load[x]] = ("loha", (
                lora[hada_keys['w1_a']], lora[hada_keys['w1_b']], 
                alpha,
                lora[hada_keys['w2_a']], lora[hada_keys['w2_b']],
                hada_t1, hada_t2
            ))
            add_to_loaded([hada_keys['w1_a'], hada_keys['w1_b'], 
                          hada_keys['w2_a'], hada_keys['w2_b']])

        # Handle LoKR (LoRA with Kronecker Product)
        lokr_components = {}
        lokr_keys = {
            'w1': f"{x}.lokr_w1",
            'w2': f"{x}.lokr_w2",
            'w1_a': f"{x}.lokr_w1_a",
            'w1_b': f"{x}.lokr_w1_b",
            'w2_a': f"{x}.lokr_w2_a",
            'w2_b': f"{x}.lokr_w2_b",
            't2': f"{x}.lokr_t2"
        }

        for key, lora_key in lokr_keys.items():
            if lora_key in lora.keys():
                lokr_components[key] = lora[lora_key]
                loaded_keys.add(lora_key)

        if any(lokr_components.values()):
            patch_dict[to_load[x]] = ("lokr", (
                lokr_components.get('w1'), lokr_components.get('w2'),
                alpha,
                lokr_components.get('w1_a'), lokr_components.get('w1_b'),
                lokr_components.get('w2_a'), lokr_components.get('w2_b'),
                lokr_components.get('t2')
            ))

        # Handle GLoRA
        glora_keys = {
            'a1': f"{x}.a1.weight",
            'a2': f"{x}.a2.weight",
            'b1': f"{x}.b1.weight",
            'b2': f"{x}.b2.weight"
        }

        if glora_keys['a1'] in lora:
            patch_dict[to_load[x]] = ("glora", (
                lora[glora_keys['a1']], lora[glora_keys['a2']],
                lora[glora_keys['b1']], lora[glora_keys['b2']],
                alpha
            ))
            add_to_loaded(glora_keys.values())

        # Handle normalization and diff weights
        for weight_type in ['norm', 'diff']:
            w_name = f"{x}.w_{weight_type}" if weight_type == 'norm' else f"{x}.{weight_type}"
            b_name = f"{x}.b_{weight_type}" if weight_type == 'norm' else f"{x}.{weight_type}_b"
            
            w = lora.get(w_name)
            b = lora.get(b_name)
            
            if w is not None:
                loaded_keys.add(w_name)
                patch_dict[to_load[x]] = ("diff", (w,))
                
                if b is not None:
                    loaded_keys.add(b_name)
                    bias_key = f"{to_load[x][:-len('.weight')]}.bias"
                    patch_dict[bias_key] = ("diff", (b,))

    # Collect remaining unprocessed weights
    remaining_dict = {k: v for k, v in lora.items() if k not in loaded_keys}
    return patch_dict, remaining_dict
