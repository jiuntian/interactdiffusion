import argparse
import random

from transformers import CLIPModel, CLIPProcessor
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from functools import partial
import os
import pickle

import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from inference import load_ckpt, prepare_batch, alpha_generator, set_alpha_scale
import inference

device = "cuda"
inference.device = device

version = "openai/clip-vit-large-patch14"
prepare_batch_model = CLIPModel.from_pretrained(version).to(device)
prepare_batch_processor = CLIPProcessor.from_pretrained(version)

inference.clip_text_feature_dict = inference.load_clip_text_cache(device)

@torch.no_grad()
def run_batch(meta, config, starting_noise=None):
    # - - - - - prepare models - - - - - #
    print(f"Loading ckpt from {meta[0]['ckpt']}")
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta[0]["ckpt"])

    if args.half:
        model = model.half()
        autoencoder = autoencoder.half()
        text_encoder = text_encoder.half()
        diffusion = diffusion.half()


    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)
    if args.no_overwrite:
        meta = [m for m in meta if not os.path.exists(os.path.join(args.folder, m["save_folder_name"], m['file_name']))]

    for i in tqdm(range(len(meta))):
        # - -` - - - prepare batch - - - - - #
        batch = prepare_batch(meta[i], config.batch_size, model=prepare_batch_model, processor=prepare_batch_processor,
                              device=device, half=args.half)

        # - - - - - generate prompt context - - - - - #
        context = text_encoder.encode([meta[i]["prompt"]] * config.batch_size)
        if args.negative_prompt is not None:
            uc = text_encoder.encode(config.batch_size * [args.negative_prompt])
        else:
            uc = text_encoder.encode(config.batch_size * [""])

        # - - - - - sampler - - - - - #
        alpha_generator_func = partial(alpha_generator, type=meta[i].get("alpha_type"))
        if config.no_plms:
            sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                                  set_alpha_scale=set_alpha_scale)
            steps = 250
        else:
            sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                                  set_alpha_scale=set_alpha_scale)
            steps = 50

            # - - - - - inpainting related - - - - - #
        inpainting_mask = z0 = None  # used for replacing known region in diffusion process
        inpainting_extra_input = None  # used as model input

            # - - - - - input for interactdiffusion - - - - - #
        grounding_input = grounding_tokenizer_input.prepare(batch)
        grounding_extra_input = None
        if grounding_downsampler_input != None:
            grounding_extra_input = grounding_downsampler_input.prepare(batch)

        input = dict(
            x=starting_noise,
            timesteps=None,
            context=context,
            grounding_input=grounding_input,
            inpainting_extra_input=inpainting_extra_input,
            grounding_extra_input=grounding_extra_input,
        )

        # - - - - - start sampling - - - - - #
        shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

        samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                      mask=inpainting_mask, x0=z0)
        samples_fake = autoencoder.decode(samples_fake)

        # - - - - - save - - - - - #
        output_folder = os.path.join(args.folder, meta[i]["save_folder_name"])
        os.makedirs(output_folder, exist_ok=True)

        start = len(os.listdir(output_folder))
        image_ids = list(range(start, start + config.batch_size))
        
        for image_id, sample in zip(image_ids, samples_fake):
            img_name = meta[i]['file_name']
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(os.path.join(output_folder, img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="generation_samples", help="root folder for output")

    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,
                        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits,'
                                ' cropped, worst quality, low quality',
                        help="negative prompt")
    parser.add_argument("--no-overwrite", action="store_true", help="do not overwrite")
    parser.add_argument("--seed", type=int, default=489)
    parser.add_argument("--scheduled-sampling", type=float, default=1.0)
    parser.add_argument("--res", type=str, default="hico", choices=["hico", "2in1"])
    parser.add_argument("--half", action='store_true', help="use 16 bit")
    args = parser.parse_args()

    assert args.batch_size == 1, 'now only support bs=1, because every image saved with same name'

    if args.res == "hico":
        res_path = "DATA/hico_det_test.pkl"
    else:
        raise ValueError(f"invalid res type: {args.res}")
    print(f"Loading res type {args.res} from {res_path}")
    res = pickle.load(open(res_path, "rb"))

    meta_list_new = []
    for r in res:  # [:2]
        meta_list_new.append(dict(
            ckpt="ckpt.pth",
            prompt=r["prompt"],
            subject_phrases=r["subject_phrases"],
            object_phrases=r["object_phrases"],
            action_phrases=r['action_phrases'],
            subject_boxes=r['subject_boxes'],
            object_boxes=r['object_boxes'],
            alpha_type=[args.scheduled_sampling, 0.0, 1-args.scheduled_sampling],
            save_folder_name=r['save_folder_name'],
            img_id=r["img_id"],
            file_name=r['file_name']
        ))
    print(f"scheduled sampling: {args.scheduled_sampling}, seed: {args.seed}"
          f" precision:{'half' if args.half else 'full'}")
    assert 1 >= args.scheduled_sampling >= 0, "scheduled sampling must be within 0 to 1"

    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    starting_noise = None

    # ------------- seeding -------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_batch(meta_list_new, args, starting_noise)
