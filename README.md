# Perturbed-Attention Guidance for ComfyUI / SD WebUI (A1111 and Forge)

Implementation of [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (D. Ahn et al.)](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/) as an extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and SD WebUI ([A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge))

Works with SD1.5, SDXL and Stable Cascade.

> [!NOTE]
> Paper and demo suggest using CFG scale 4.0 with PAG scale 3.0 applied to U-Net's middle layer 0, but feel free to experiment.
>
> Sampling speed without `adaptive_scale` is similar to Self-Attention Guidance (x0.6 of usual it/s).

## Installation

### ComfyUI

Basic PAG node is now included into ComfyUI - you don't have to install this extension unless you want to mess with additional parameters.

![comfyui-node-basic](examples/comfyui-node-basic.png)

To install advanced PAG node from this repo, you can either:

- `git clone https://github.com/pamparamm/sd-perturbed-attention.git` into `ComfyUI/custom-nodes/` folder.

- Install it via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) (search for custom node named "Perturbed-Attention Guidance").

![comfyui-node-advanced](examples/comfyui-node-advanced.png)

### SD WebUI (A1111 and Forge)

`git clone https://github.com/pamparamm/sd-perturbed-attention.git` into `stable-diffusion-webui/extensions/` folder.

> [!WARNING]
> Extension for A1111 WebUI is still under development - only `Scale` works for now.

![webui-script](examples/webui-script.png)

> [!NOTE]
> You can override `CFG Scale` and `PAG Scale` for Hires. fix by opening/enabling `Override for Hires. fix` tab.
> To disable PAG during Hires. fix, set `PAG Scale` under Override to 0.

As an alternative for A1111 WebUI you can use PAG implementation from [sd-webui-incantations](https://github.com/v0xie/sd-webui-incantations) extension.

## Parameters

- `scale`: PAG scale, has some resemblance to CFG scale - higher values can both increase structural coherence of the image and oversaturate/fry it entirely.
- `adaptive_scale`: PAG dampening factor, it penalizes PAG during late denoising stages, resulting in overall speedup: 0.0 means no penalty and 1.0 completely removes PAG.
- `unet_block`: Part of U-Net to which PAG is applied, original paper suggests to use `middle`.
- `unet_block_id`: Id of U-Net layer in a selected block to which PAG is applied. PAG can be applied only to layers containing Self-attention blocks.