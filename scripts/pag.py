try:
    import pag_nodes

    if pag_nodes.BACKEND == "Forge":
        import gradio as gr

        from modules import scripts
        from modules.ui_components import InputAccordion

        opPerturbedAttention = pag_nodes.PerturbedAttention()

        class PerturbedAttentionScript(scripts.Script):
            def title(self):
                return "Perturbed-Attention Guidance"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)
                    scale = gr.Slider(label="PAG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                    with gr.Row():
                        rescale_pag = gr.Slider(label="Rescale PAG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        rescale_mode = gr.Dropdown(choices=["full", "partial"], value="full", label="Rescale Mode")
                    adaptive_scale = gr.Slider(label="Adaptive Scale", minimum=0.0, maximum=1.0, step=0.001, value=0.0)
                    with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                        hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
                        hr_scale = gr.Slider(label="PAG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                        with gr.Row():
                            hr_rescale_pag = gr.Slider(label="Rescale PAG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            hr_rescale_mode = gr.Dropdown(choices=["full", "partial"], value="full", label="Rescale Mode")
                        hr_adaptive_scale = gr.Slider(label="Adaptive Scale", minimum=0.0, maximum=1.0, step=0.001, value=0.0)
                    with gr.Row():
                        block = gr.Dropdown(choices=["input", "middle", "output"], value="middle", label="U-Net Block")
                        block_id = gr.Number(label="U-Net Block Id", value=0, precision=0, minimum=0)
                        block_list = gr.Text(label="U-Net Block List")
                    with gr.Row():
                        sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                        sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=-1.0)

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="pag_enabled" in p)),
                        (scale, "pag_scale"),
                        (rescale_pag, "pag_rescale"),
                        (rescale_mode, lambda p: gr.Dropdown.update(value=p.get("pag_rescale_mode", "full"))),
                        (adaptive_scale, "pag_adaptive_scale"),
                        (hr_override, lambda p: gr.Checkbox.update(value="hr_override" in p)),
                        (hr_cfg, "pag_hr_cfg"),
                        (hr_scale, "pag_hr_scale"),
                        (hr_rescale_pag, "pag_hr_rescale"),
                        (hr_rescale_mode, lambda p: gr.Dropdown.update(value=p.get("pag_hr_rescale_mode", "full"))),
                        (hr_adaptive_scale, "pag_hr_adaptive_scale"),
                        (block, lambda p: gr.Dropdown.update(value=p.get("pag_block", "middle"))),
                        (block_id, "pag_block_id"),
                        (block_list, lambda p: gr.Text.update(value=p.get("pag_block_list", ""))),
                        (sigma_start, "pag_sigma_start"),
                        (sigma_end, "pag_sigma_end"),
                    )

                return enabled, scale, rescale_pag, rescale_mode, adaptive_scale, block, block_id, block_list, hr_override, hr_cfg, hr_scale, hr_rescale_pag, hr_rescale_mode, hr_adaptive_scale, sigma_start, sigma_end

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                (
                    enabled,
                    scale,
                    rescale_pag,
                    rescale_mode,
                    adaptive_scale,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_rescale_pag,
                    hr_rescale_mode,
                    hr_adaptive_scale,
                    sigma_start,
                    sigma_end,
                ) = script_args

                if not enabled:
                    return

                unet = p.sd_model.forge_objects.unet

                hr_enabled = getattr(p, "enable_hr", False)

                if hr_enabled and p.is_hr_pass and hr_override:
                    p.cfg_scale_before_hr = p.cfg_scale
                    p.cfg_scale = hr_cfg
                    unet = opPerturbedAttention.patch(unet, hr_scale, hr_adaptive_scale, block, block_id, sigma_start, sigma_end, hr_rescale_pag, hr_rescale_mode, block_list)[0]
                else:
                    unet = opPerturbedAttention.patch(unet, scale, adaptive_scale, block, block_id, sigma_start, sigma_end, rescale_pag, rescale_mode, block_list)[0]

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        pag_enabled=enabled,
                        pag_scale=scale,
                        pag_rescale=rescale_pag,
                        pag_rescale_mode=rescale_mode,
                        pag_adaptive_scale=adaptive_scale,
                        pag_block=block,
                        pag_block_id=block_id,
                        pag_block_list=block_list,
                    )
                )
                if hr_enabled:
                    p.extra_generation_params["pag_hr_override"] = hr_override
                    if hr_override:
                        p.extra_generation_params.update(
                            dict(
                                pag_hr_cfg=hr_cfg,
                                pag_hr_scale=hr_scale,
                                pag_hr_rescale=hr_rescale_pag,
                                pag_hr_rescale_mode=hr_rescale_mode,
                                pag_hr_adaptive_scale=hr_adaptive_scale,
                            )
                        )
                if sigma_start >= 0 or sigma_end >= 0:
                    p.extra_generation_params.update(
                        dict(
                            pag_sigma_start=sigma_start,
                            pag_sigma_end=sigma_end,
                        )
                    )

                return

            def post_sample(self, p, ps, *script_args):
                (
                    enabled,
                    scale,
                    rescale_pag,
                    rescale_mode,
                    adaptive_scale,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_rescale_pag,
                    hr_rescale_mode,
                    hr_adaptive_scale,
                    sigma_start,
                    sigma_end,
                ) = script_args

                if not enabled:
                    return

                hr_enabled = getattr(p, "enable_hr", False)

                if hr_enabled and hr_override:
                    p.cfg_scale = p.cfg_scale_before_hr

                return

except ImportError:
    pass
