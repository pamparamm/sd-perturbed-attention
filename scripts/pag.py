try:
    from scripts import pag_nodes

    if pag_nodes.BACKEND == "Forge":
        import gradio as gr

        from modules import scripts

        opPerturbedAttention = pag_nodes.PerturbedAttention()

        class PerturbedAttentionScript(scripts.Script):
            def title(self):
                return "Perturbed-Attention Guidance"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)
                    with gr.Row():
                        scale = gr.Slider(label="Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                        adaptive_scale = gr.Slider(label="Adaptive Scale", minimum=0.0, maximum=1.0, step=0.001, value=0.0)
                    with gr.Row():
                        block = gr.Dropdown(choices=["input", "middle", "output"], value="middle", label="U-Net Block")
                        block_id = gr.Number(label="U-Net Block Id", value=0, precision=0, minimum=0)

                return enabled, scale, adaptive_scale, block, block_id

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                enabled, scale, adaptive_scale, block, block_id = script_args

                if not enabled:
                    return

                unet = p.sd_model.forge_objects.unet

                unet = opPerturbedAttention.patch(unet, scale, adaptive_scale, block, block_id)[0]

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        pag_enabled=enabled,
                        pag_scale=scale,
                        pag_adaptive_scale = adaptive_scale,
                        pag_block=block,
                        pag_block_id=block_id,
                    )
                )

                return

except ImportError:
    pass
