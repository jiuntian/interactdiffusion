
# InteractDiffusion: Interaction-Control for Text-to-Image Diffusion Model

[Jiun Tian Hoe](https://jiuntian.com/), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/),
[Chee Seng Chan](http://cs-chan.com), [Yap Peng Tan](https://personal.ntu.edu.sg/eyptan/),
[Weipeng Hu](https://scholar.google.com/citations?user=zo6ni_gAAAAJ)

[Project Page](https://jiuntian.github.io/interactdiffusion) |
 [Paper](https://arxiv.org/abs/2312.05849) |
 [WebUI](https://github.com/jiuntian/sd-webui-interactdiffusion) |
 [Demo (coming soon)](https://huggingface.co/spaces/interactdiffusion/demo) |
 [Video](https://www.youtube.com/watch?v=Uunzufq8m6Y)

[![Paper](https://img.shields.io/badge/cs.CV-arxiv:2312.05849-B31B1B.svg)](https://arxiv.org/abs/2312.05849)
[![Page Views Count](https://badges.toozhao.com/badges/01HH1JE53YX5TDDDDCG6PXY8WQ/blue.svg)](https://badges.toozhao.com/stats/01HH1JE53YX5TDDDDCG6PXY8WQ "Get your own page views count badge on badges.toozhao.com")

![Teaser figure](docs/static/res/teaser.jpg)

<!-- [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Uunzufq8m6Y/0.jpg)](https://youtu.be/Uunzufq8m6Y) -->

- Existing methods lack ability to control the interactions between objects in the generated content.
- We propose a pluggable interaction control model, called InteractDiffusion that extends existing pre-trained T2I diffusion models to enable them being better conditioned on interactions.

## News

- **[2024.2.27]** InteractionDiffusion paper is accepted at CVPR 2024.
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

## Extension for AutomaticA111's Stable Diffusion WebUI

We develop an AutomaticA111's Stable Diffuion WebUI extension to allow the use of InteractDiffusion over existing SD models. Check out the plugin at [sd-webui-interactdiffusion](https://github.com/jiuntian/sd-webui-interactdiffusion). Note that it is still on `alpha` version.

### Gallery

Some examples generated with InteractDiffusion, together with other DreamBooth and LoRA models.
&nbsp;| &nbsp;| &nbsp;| &nbsp;
--- | --- | --- | ---
![image (7)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/e4ff1279-1b08-41c9-9ea3-45ec3667115e) | ![image (5)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/dfd254ea-f6fb-4fc4-9fe6-8222fe47ee12) | ![image (6)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/a6df1288-3315-4738-9db8-d9cb9bd01038) | ![image (4)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/1766e775-ce6c-4705-a376-4aa8e62bcceb)
![cuteyukimix_1](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/1416f2b6-4907-4ac7-bb03-b5d2b5adcd91)|![cuteyukimix_7](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/7b619e4e-7d0b-4989-85f9-422fbd6a6319)|![darksushimix_1](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/2b81abe3-a39a-4db8-9e7a-63336f96d7e3)|![toonyou_6](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/ce027fac-7840-44cc-9f69-0bdeef5da1da)
![image (8)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/0bc70ee4-9f84-4340-994c-fbde99a17062)|![cuteyukimix_4](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/0d12f242-cc90-4871-8d2c-02f7c36c70cf)|![darksushimix_5](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/cd716268-92d2-48fa-bbc5-a291c80f7f9a)|![rcnzcartoon_1](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/ce8c33f1-62fd-4c44-ae76-d5b70b1f05f5)

## Inference

🚧 Code is working on progress and will be open soon. 🏗️  🔨 Please stay tuned!

## Training

1. Prepare the necessary dataset and pretrained models, see [DATA](DATA/readme.md)
2. Run the following command:

      ```bash
      CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py --yaml_file configs/hoi_hico_text.yaml --ckpt <existing_gligen_checkpoint> --name test --batch_size=4 --gradient_accumulation_step 2 --total_iters 500000 --amp true --disable_inference_in_training true --official_ckpt_name <existing SD v1.4/v1.5 checkpoint>
      ```

## TODO

- [x] Code Release
- [x] HuggingFace demo
- [x] WebUI extension
- [ ] Diffuser

## Citation

```bibtex
@inproceedings{hoe2023interactdiffusion,
      title={InteractDiffusion: Interaction Control in Text-to-Image Diffusion Models}, 
      author={Jiun Tian Hoe and Xudong Jiang and Chee Seng Chan and Yap-Peng Tan and Weipeng Hu},
      year={2024},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

## Acknowledgement

This work is developed based on the codebase of [GLIGEN](https://github.com/gligen/GLIGEN) and [LDM](https://github.com/CompVis/latent-diffusion).
