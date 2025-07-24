# Various Guidance implementations for ComfyUI / SD WebUI (reForge)

Implementation of

- Perturbed-Attention Guidance from [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (D. Ahn et al.)](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/)
- [Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention (Susung Hong)](https://arxiv.org/abs/2408.00760)
- Sliding Window Guidance from [The Unreasonable Effectiveness of Guidance for Diffusion Models (Kaiser et al.)](https://arxiv.org/abs/2411.10257)
- [PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity](https://cubeyoung.github.io/pladis-proejct/) (ComfyUI-only)
- [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://arxiv.org/abs/2505.21179)
- [Token Perturbation Guidance for Diffusion Models](https://arxiv.org/abs/2506.10036) (ComfyUI-only)

as an extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [SD WebUI (reForge)](https://github.com/Panchovix/stable-diffusion-webui-reForge).

Works with SD1.5 and SDXL.

> [!NOTE]
> `Normalized Attention Guidance` node has its own description inside ComfyUI - you can click on the node and press the button with an encircled question mark.

> [!NOTE]
> PLADIS isn't working properly right now - for some reason it produces messy images when applied to all cross-attention layers. I'll try to investigate the problem, but it wouldn't be easy considering the lack of official implementation + various typos in the original paper. For debug purposes, I might implement PLADIS as a diffusers pipeline to see if the problem still occurs.
>
> Any help is welcomed!

## Installation

### ComfyUI

You can either:

- `git clone https://github.com/pamparamm/sd-perturbed-attention.git` into `ComfyUI/custom-nodes/` folder.

- Install it via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) (search for custom node named "Perturbed-Attention Guidance").

- Install it via [comfy-cli](https://comfydocs.org/comfy-cli/getting-started) with `comfy node registry-install sd-perturbed-attention`

### SD WebUI (reForge)

`git clone https://github.com/pamparamm/sd-perturbed-attention.git` into `stable-diffusion-webui-forge/extensions/` folder.

### SD WebUI (Auto1111)

As an alternative for A1111 WebUI you can use PAG implementation from [sd-webui-incantations](https://github.com/v0xie/sd-webui-incantations) extension.

## Guidance Nodes/Scripts

### ComfyUI

![comfyui-node-pag-basic](res/comfyui-node-pag-basic.png)

![comfyui-node-pag-advanced](res/comfyui-node-pag-advanced.png)

![comfyui-node-seg](res/comfyui-node-seg.png)

### SD WebUI (reForge)

![forge-pag](res/forge-pag.png)

![forge-seg](res/forge-seg.png)

![forge-nag](res/forge-nag.png)

> [!NOTE]
> You can override `CFG Scale` and `PAG Scale`/`SEG Scale`/`NAG Scale` for Hires. fix by opening/enabling `Override for Hires. fix` tab.
> To disable guidance during Hires. fix, you can set the scale under Override to 0 or use the `HRFix Off` option in the `Hires Fix Mode` dropdown for widgets that support it.

### Inputs

- `scale`: Guidance scale, higher values can both increase structural coherence of an image and oversaturate/fry it entirely.
- `adaptive_scale` (PAG only): PAG dampening factor, it penalizes PAG during late denoising stages, resulting in overall speedup: 0.0 means no penalty and 1.0 completely removes PAG.
- `blur_sigma` (SEG only): Normal deviation of Gaussian blur, higher values increase "clarity" of an image. Negative values set `blur_sigma` to infinity.
- `unet_block`: Part of U-Net to which Guidance is applied, original paper suggests to use `middle`.
- `unet_block_id`: Id of U-Net layer in a selected block to which Guidance is applied. Guidance can be applied only to layers containing Self-attention blocks.
- `sigma_start` / `sigma_end`: Guidance will be active only between `sigma_start` and `sigma_end`. Set both values to negative to disable this feature.
- `rescale`: Acts similar to RescaleCFG node - it prevents over-exposure on high `scale` values. Based on Algorithm 2 from [Common Diffusion Noise Schedules and Sample Steps are Flawed (Lin et al.)](https://arxiv.org/abs/2305.08891). Set to 0 to disable this feature.
- `rescale_mode`:
  - `full` - takes into account both CFG and Guidance.
  - `partial` - depends only on Guidance.
  - `snf` - Saliency-adaptive Noise Fusion from [High-fidelity Person-centric Subject-to-Image Synthesis (Wang et al.)](https://arxiv.org/abs/2311.10329). Should increase image quality on high guidance scales. Ignores `rescale` value.
- `unet_block_list`: Optional input, replaces both `unet_block` and `unet_block_id` and allows you to select multiple U-Net layers separated with commas. SDXL U-Net has multiple indices for layers, you can specify them by using dot symbol (if not specified, Guidance will be applied to the whole layer). Example value: `m0,u0.4` (it applies Guidance to middle block 0 and to output block 0 with index 4)
  - In terms of U-Net `d` means `input`, `m` means `middle` and `u` means `output`.
  - SD1.5 U-Net has layers `d0`-`d5`, `m0`, `u0`-`u8`.
  - SDXL U-Net has layers `d0`-`d3`, `m0`, `u0`-`u5`. In addition, each block except `d0` and `d1` has `0-9` index values (like `m0.7` or `u0.4`). `d0` and `d1` have `0-1` index values.
  - Supports block ranges (`d0-d3` corresponds to `d0,d1,d2,d3`) and index value ranges (`d2.2-9` corresponds to all index values of `d2` with the exclusion of `d2.0` and `d2.1`).
- `hr_mode`: Controls when guidance is active during generation
  - `Both` - Apply guidance in both base generation and Hires. fix (default)
  - `HRFix Off` - Only apply guidance during base generation, disable during Hires. fix
  - `HRFix Only` - Only apply guidance during Hires. fix, disable during base generation

### NAG-specific Inputs

- `negative`: NAG negative prompt. When empty, uses the main negative prompt.
- `tau`: Normalization threshold. Higher values increase the impact of the scale parameter.
- `alpha`: Linear interpolation between original (at alpha=0) and NAG (at alpha=1) results.

## ComfyUI TensorRT PAG (Experimental)

To use PAG together with [ComfyUI_TensorRT](https://github.com/comfyanonymous/ComfyUI_TensorRT), you'll need to:

0. Have 24GB of VRAM.
1. Build static/dynamic TRT engine of a desired model.
2. Build static/dynamic TRT engine of the same model with the same TRT parameters, but with fixed PAG injection in selected UNET blocks (`TensorRT Attach PAG` node).
3. Use `TensorRT Perturbed-Attention Guidance` node with two model inputs: one for base engine and one for PAG engine.

![trt-engines](res/trt-engines.png)

![trt-inference](res/trt-inference.png)
