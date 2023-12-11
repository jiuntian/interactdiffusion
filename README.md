
# InteractDiffusion: Interaction-Control for Text-to-Image Diffusion Model

[Jiun Tian Hoe](https://jiuntian.com/), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/),
[Chee Seng Chan](http://cs-chan.com), [Yap Peng Tan](https://personal.ntu.edu.sg/eyptan/),
[Weipeng Hu](https://scholar.google.com/citations?user=zo6ni_gAAAAJ)

[Project Page](https://jiuntian.github.io/interactdiffusion) |
 [Paper](https://arxiv.org/abs/?) |
 [WebUI](https://github.com/jiuntian/sd-webui-interactdiffusion) |
 [Demo (coming soon)](https://huggingface.co/spaces/interactdiffusion/demo) |
 [Video](https://www.youtube.com/watch?v=Uunzufq8m6Y)

[![Page Views Count](https://badges.toozhao.com/badges/01HH1JE53YX5TDDDDCG6PXY8WQ/blue.svg)](https://badges.toozhao.com/stats/01HH1JE53YX5TDDDDCG6PXY8WQ "Get your own page views count badge on badges.toozhao.com")

![Teaser figure](docs/static/res/teaser.jpg)

<!-- [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Uunzufq8m6Y/0.jpg)](https://youtu.be/Uunzufq8m6Y) -->

- Existing methods lack ability to control the interactions between objects in the generated content.
- We propose a pluggable interaction control model, called InteractDiffusion that extends existing pre-trained T2I diffusion models to enable them being better conditioned on interactions.

## News

- **[2023.12.12]** InteractionDiffusion paper is released. WebUI of InteractDiffusion is available as *alpha* version.

## Download InteractDiffusion models

We provide three checkpoints with different training strategies.
| Version | Dataset    | SD |Download |
|---------|------------|----|---------|
| v1.0 | HICO-DET                 | v1.4| [HF Hub](https://huggingface.co/jiuntian/interactiondiffusion-weight/blob/main/interact-diffusion-v1.pth) |
| v1.1 | HICO-DET                 | v1.5| [HF Hub](https://huggingface.co/jiuntian/interactiondiffusion-weight/blob/main/interact-diffusion-v1-1.pth) |
| v1.2 | HICO-DET + VisualGenome  | v1.5| [HF Hub](https://huggingface.co/jiuntian/interactiondiffusion-weight/blob/main/interact-diffusion-v1-2.pth) |

Note that the experimental results in our paper is referring to v1.0.

- v1.0 is based on Stable Diffusion v1.4 and GLIGEN. We train at batch size of 16 for 250k steps on HICO-DET. **Our paper is based on this.**
- v1.1 is based on Stable Diffusion v1.5 and GLIGEN. We train at batch size of 32 for 250k steps on HICO-DET.
- v1.1 is based on InteractDiffusion v1.1. We train further at batch size of 32 for 172.5k steps on HICO-DET and VisualGenome.

## Extension for AutomaticA111's Stable Diffuion WebUI

We develop an AutomaticA111's Stable Diffuion WebUI extension to allow the use of InteractDiffusion over existing SD models. Check out the plugin at [sd-webui-interactdiffusion](https://github.com/jiuntian/sd-webui-interactdiffusion). Note that it is still on `alpha`.

## Inference and Training Code

🚧 Code is working on progress and will be open soon. 🏗️  🔨 Please stay tuned!

## TODO

- [ ] Code Release
- [ ] HuggingFace demo
- [x] WebUI extension

## Citation

```bibtex
???
```
