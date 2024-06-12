
# InteractDiffusion: Interaction-Control for Text-to-Image Diffusion Model

[Jiun Tian Hoe](https://jiuntian.com/), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/),
[Chee Seng Chan](http://cs-chan.com), [Yap Peng Tan](https://personal.ntu.edu.sg/eyptan/),
[Weipeng Hu](https://scholar.google.com/citations?user=zo6ni_gAAAAJ)

[Project Page](https://jiuntian.github.io/interactdiffusion) |
 [Paper](https://arxiv.org/abs/2312.05849) |
 [WebUI](https://github.com/jiuntian/sd-webui-interactdiffusion) |
 [Demo](https://huggingface.co/spaces/interactdiffusion/interactdiffusion) |
 [Video](https://www.youtube.com/watch?v=Uunzufq8m6Y) |
 [Diffuser](https://huggingface.co/interactdiffusion/diffusers-v1-2) |
 [Colab](https://colab.research.google.com/drive/1Bh9PjfTylxI2rbME5mQJtFqNTGvaghJq?usp=sharing)

[![Paper](https://img.shields.io/badge/cs.CV-arxiv:2312.05849-B31B1B.svg)](https://arxiv.org/abs/2312.05849)
[![Page Views Count](https://badges.toozhao.com/badges/01HH1JE53YX5TDDDDCG6PXY8WQ/blue.svg)](https://badges.toozhao.com/stats/01HH1JE53YX5TDDDDCG6PXY8WQ "Get your own page views count badge on badges.toozhao.com")
[![Hugging Face](https://img.shields.io/badge/InteractDiffusion-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/interactdiffusion/interactdiffusion)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Bh9PjfTylxI2rbME5mQJtFqNTGvaghJq?usp=sharing)

![Teaser figure](docs/static/res/teaser.jpg)

<!-- [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Uunzufq8m6Y/0.jpg)](https://youtu.be/Uunzufq8m6Y) -->

- Existing methods lack ability to control the interactions between objects in the generated content.
- We propose a pluggable interaction control model, called InteractDiffusion that extends existing pre-trained T2I diffusion models to enable them being better conditioned on interactions.

## News

- **[2024.3.13]** Diffusers code is available at [here](https://huggingface.co/interactdiffusion/diffusers-v1-2).
- **[2024.3.8]** Demo is available at [Huggingface Spaces](https://huggingface.co/spaces/interactdiffusion/interactdiffusion).
- **[2024.3.6]** Code is released.
- **[2024.2.27]** InteractionDiffusion paper is accepted at CVPR 2024.
- **[2023.12.12]** InteractionDiffusion paper is released. WebUI of InteractDiffusion is available as *alpha* version.

## Results

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">Interaction Controllability</th>
    <th rowspan="2">FID</th>
    <th rowspan="2">KID</th>
  </tr>
  <tr>
    <th>Tiny</th>
    <th>Large</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>v1.0</td>
    <td>29.53</td>
    <td>31.56</td>
    <td>18.69</td>
    <td>0.00676</td>
  </tr>
  <tr>
    <td>v1.1</td>
    <td>30.20</td>
    <td>31.96</td>
    <td>17.90</td>
    <td>0.00635</td>
  </tr>
  <tr>
    <td>v1.2</td>
    <td>30.73</td>
    <td>33.10</td>
    <td>17.32</td>
    <td>0.00585</td>
  </tr>
</tbody>
</table>

  Interaction Controllability is measured using FGAHOI detection score. In this table, we measure the Full subset in Default setting on Swin-Tiny and Swin-Large backbone. More details on the protocol is in the paper.

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

## Diffusers
```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "interactdiffusion/diffusers-v1-2",
    trust_remote_code=True,
    variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

images = pipeline(
    prompt="a person is feeding a cat",
    interactdiffusion_subject_phrases=["person"],
    interactdiffusion_object_phrases=["cat"],
    interactdiffusion_action_phrases=["feeding"],
    interactdiffusion_subject_boxes=[[0.0332, 0.1660, 0.3359, 0.7305]],
    interactdiffusion_object_boxes=[[0.2891, 0.4766, 0.6680, 0.7930]],
    interactdiffusion_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
    ).images

images[0].save('out.jpg')
```

## Reproduce & Evaluate

1. Change `ckpt.pth` in interence_batch.py to selected checkpoint.
2. Made inference on InteractDiffusion to synthesis the test set of HICO-DET based on the ground truth.

      ```bash
      python inference_batch.py --batch_size 1 --folder generated_output --seed 489 --scheduled-sampling 1.0 --half
      ```
  
3. Setup FGAHOI at `../FGAHOI`. See [FGAHOI repo](https://github.com/xiaomabufei/FGAHOI) on how to setup FGAHOI and also HICO-DET dataset in `data/hico_20160224_det`.
4. Prepare for evaluate on FGAHOI. See `id_prepare_inference.ipynb`
5. Evaluate on FGAHOI.

      ```bash
      python main.py --backbone swin_tiny --dataset_file hico --resume weights/FGAHOI_Tiny.pth --num_verb_classes 117 --num_obj_classes 80 --output_dir logs  --merge --hierarchical_merge --task_merge --eval --hoi_path data/id_generated_output --pretrain_model_path "" --output_dir logs/id-generated-output-t
      ```

6. Evaluate for FID and KID. We recommend to resize hico_det dataset to 512x512 before perform image quality evaluation, for a fair comparison. We use [torch-fidelity](https://github.com/toshas/torch-fidelity).

      ```bash
      fidelity --gpu 0 --fid --isc --kid --input2 ~/data/hico_det_test_resize  --input1 ~/FGAHOI/data/data/id_generated_output/images/test2015
      ```

7. This should provide a brief overview of how the evaluation process works.

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
- [x] Diffuser

## Citation

```bibtex
@InProceedings{Hoe_2024_CVPR,
    author    = {Hoe, Jiun Tian and Jiang, Xudong and Chan, Chee Seng and Tan, Yap-Peng and Hu, Weipeng},
    title     = {InteractDiffusion: Interaction Control in Text-to-Image Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {6180-6189}
}
```

## Acknowledgement

This work is developed based on the codebase of [GLIGEN](https://github.com/gligen/GLIGEN) and [LDM](https://github.com/CompVis/latent-diffusion).
