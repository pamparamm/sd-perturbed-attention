# Various Guidance implementations for ComfyUI / SD WebUI (reForge)

Implementation of

- Perturbed-Attention Guidance (PAG) from [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (D. Ahn et al.)](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/)
- [Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention (Susung Hong)](https://arxiv.org/abs/2408.00760)
- Sliding Window Guidance (SWG) from [The Unreasonable Effectiveness of Guidance for Diffusion Models (Kaiser et al.)](https://arxiv.org/abs/2411.10257)
- [PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity](https://cubeyoung.github.io/pladis-proejct/) (ComfyUI-only)
- [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://arxiv.org/abs/2505.21179) (ComfyUI-only, has a description inside ComfyUI)
- [Token Perturbation Guidance for Diffusion Models](https://arxiv.org/abs/2506.10036) (ComfyUI-only)
- Frequency-Decoupled Guidance (FDG) from [Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales](https://arxiv.org/abs/2506.19713) (ComfyUI-only)

as an extension for [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and [SD WebUI (reForge)](https://github.com/Panchovix/stable-diffusion-webui-reForge).

Works with SD1.5 and SDXL.

## Installation

### ComfyUI

You can either:

- `git clone https://github.com/pamparamm/sd-perturbed-attention.git` into `ComfyUI/custom-nodes/` folder.

- Install it via [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager) (search for custom node named "Perturbed-Attention Guidance").

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

> [!NOTE]
> You can override `CFG Scale` and `PAG Scale`/`SEG Scale` for Hires. fix by opening/enabling `Override for Hires. fix` tab.
> To disable PAG during Hires. fix, you can set `PAG Scale` under Override to 0.

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

## ComfyUI TensorRT PAG (Experimental)

Deprecated: [ComfyUI_TensorRT](https://github.com/comfyanonymous/ComfyUI_TensorRT) is unmaintained.


## Citation
```
@misc{ahn2025selfrectifyingdiffusionsamplingperturbedattention,
      title={Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance},
      author={Donghoon Ahn and Hyoungwon Cho and Jaewon Min and Wooseok Jang and Jungwoo Kim and SeonHwa Kim and Hyun Hee Park and Kyong Hwan Jin and Seungryong Kim},
      year={2025},
      eprint={2403.17377},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.17377},
}

@misc{hong2024smoothedenergyguidanceguiding,
      title={Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention},
      author={Susung Hong},
      year={2024},
      eprint={2408.00760},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00760},
}

@misc{adaloglou2025guidingdiffusionmodelusing,
      title={Guiding a diffusion model using sliding windows},
      author={Nikolas Adaloglou and Tim Kaiser and Damir Iagudin and Markus Kollmann},
      year={2025},
      eprint={2411.10257},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10257},
}

@misc{kim2025pladispushinglimitsattention,
      title={PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity},
      author={Kwanyoung Kim and Byeongsu Sim},
      year={2025},
      eprint={2503.07677},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.07677},
}

@misc{chen2025normalizedattentionguidanceuniversal,
      title={Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models},
      author={Dar-Yen Chen and Hmrishav Bandyopadhyay and Kai Zou and Yi-Zhe Song},
      year={2025},
      eprint={2505.21179},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.21179},
}

@misc{rajabi2025tokenperturbationguidancediffusion,
      title={Token Perturbation Guidance for Diffusion Models},
      author={Javad Rajabi and Soroush Mehraban and Seyedmorteza Sadat and Babak Taati},
      year={2025},
      eprint={2506.10036},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.10036},
}

@misc{sadat2025guidancefrequencydomainenables,
      title={Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales},
      author={Seyedmorteza Sadat and Tobias Vontobel and Farnood Salehi and Romann M. Weber},
      year={2025},
      eprint={2506.19713},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.19713},
}
```