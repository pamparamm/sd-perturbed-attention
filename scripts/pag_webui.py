import pag_utils

if pag_utils.BACKEND == "WebUI":
    from modules import shared
    from modules.hypernetworks import hypernetwork
    from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
    from modules.script_callbacks import on_cfg_denoiser, on_cfg_denoised, on_cfg_after_cfg
    from ldm.util import default

    INITIALIZED = False

    class PAGState:
        def __init__(self, *script_args):
            (
                self.enabled,
                self.scale,
                self.adaptive_scale,
                self.block,
                self.block_id,
                self.hr_override,
                self.hr_cfg,
                self.hr_scale,
                self.hr_adaptive_scale,
            ) = script_args

            _model = shared.sd_model.model.diffusion_model
            _middle_transformers = _model.middle_block[1].transformer_blocks
            self._middle_attn1 = [block.attn1 for block in _middle_transformers]
            self._middle_attn1_forward_backup = [a.forward for a in self._middle_attn1]

            self.kwargs = None

        def _attn_replace(self) -> None:
            self._middle_attn1[0].forward = self.pag_attention_forward(self._middle_attn1[0])

        def _attn_restore(self):
            self._middle_attn1[0].forward = self._middle_attn1_forward_backup[0]

        def pag_attention_forward(self, a):
            def _pag_attention_forward(x, context=None, mask=None, **kwargs):
                context = default(context, x)
                _, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
                v_in = a.to_v(context_v)
                return a.to_out(v_in)

            return _pag_attention_forward

    def register_callbacks(script):
        global INITIALIZED

        def __denoiser_callback(params: CFGDenoiserParams):
            state: PAGState = script.pag_state
            if not state.enabled:
                return

            state.kwargs = {
                "batch_size": params.text_cond.shape[0],
                "latent": params.x[: params.text_cond.shape[0]],
                "sigma": params.sigma[: params.text_cond.shape[0]],
                "text_cond": params.text_cond,
                "image_cond": params.image_cond[: params.text_cond.shape[0]],
            }

        def __denoised_callback(params: CFGDenoisedParams):
            state: PAGState = script.pag_state

            try:
                if not state.enabled:
                    return

                state.kwargs["cond_pred"] = params.x[: state.kwargs["batch_size"]]

                if shared.sd_model.model.conditioning_key == "crossattn-adm":
                    make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": c_crossattn, "c_adm": c_adm}
                else:
                    if isinstance(state.kwargs["text_cond"], dict):
                        make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
                    else:
                        make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}

                state._attn_replace()
                state.kwargs["pag_pred"] = params.inner_model(
                    state.kwargs["latent"],
                    state.kwargs["sigma"],
                    cond=make_condition_dict(state.kwargs["text_cond"], state.kwargs["image_cond"]),
                )

            finally:
                state._attn_restore()

        def __cfg_after_cfg_callback(params: AfterCFGCallbackParams):
            state: PAGState = script.pag_state

            if not state.enabled:
                return

            params.x = params.x + ((state.kwargs["cond_pred"] - state.kwargs["pag_pred"]) * float(state.scale))

        if not INITIALIZED:
            on_cfg_denoiser(__denoiser_callback)
            on_cfg_denoised(__denoised_callback)
            on_cfg_after_cfg(__cfg_after_cfg_callback)
            INITIALIZED = True
