import pag_utils

if pag_utils.BACKEND == "Forge":
    import pag_nodes

    opPerturbedAttention = pag_nodes.PerturbedAttention()

    def patch_unet(p, is_overriding_hr=False, *script_args):
        enabled, scale, adaptive_scale, block, block_id, hr_override, hr_cfg, hr_scale, hr_adaptive_scale = script_args

        unet = p.sd_model.forge_objects.unet

        if is_overriding_hr:
            unet = opPerturbedAttention.patch(unet, hr_scale, hr_adaptive_scale, block, block_id)[0]
        else:
            unet = opPerturbedAttention.patch(unet, scale, adaptive_scale, block, block_id)[0]

        p.sd_model.forge_objects.unet = unet
