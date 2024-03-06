import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
import numpy as np
import clip
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# device = "cpu"
device = "cuda"
clip_text_feature_dict = dict()


def load_clip_text_cache(device):
    clip_text_feature_dict = torch.load('../clip_phrases_feature_cache.pth', map_location=device)
    return clip_text_feature_dict


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1, 0, 0]

    assert len(type) == 3
    assert type[0] + type[1] + type[2] == 1

    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1] * stage0_length + decay_alphas + [0] * stage2_length

    assert len(alphas) == length

    return alphas


def load_ckpt(ckpt_path):
    saved_ckpt = torch.load(ckpt_path, map_location=device)
    config = saved_ckpt["config_dict"]["_content"]
    config['text_encoder'].update({
        'params': {
            'device': device
        }
    })

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict(saved_ckpt['model'])
    autoencoder.load_state_dict(saved_ckpt["autoencoder"])
    text_encoder.load_state_dict(saved_ckpt["text_encoder"])
    diffusion.load_state_dict(saved_ckpt["diffusion"])

    return model, autoencoder, text_encoder, diffusion, config


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x @ torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False, device=device):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)  # we use our own preprocessing without center_crop
        inputs['input_ids'] = torch.tensor([[0, 1, 2, 3]]).to(device)  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds
        if which_layer_image == 'after_reproject':
            feature = project(feature, torch.load('projection_matrix').to(device).T).squeeze(0)
            feature = (feature / feature.norm()) * 28.7
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        if input not in clip_text_feature_dict:
            print(f"CLIP feature for phrase {input} not found, creating")
            inputs = processor(text=input, return_tensors="pt", padding=True)
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['pixel_values'] = torch.ones(1, 3, 224, 224).to(device)  # placeholder
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            outputs = model(**inputs)
            if which_layer_text == 'before':
                feature = outputs.text_model_output.pooler_output
                clip_text_feature_dict[input] = feature
        else:
            feature = clip_text_feature_dict[input]

    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1, max_objs)
    if has_mask == None:
        return mask

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0, idx] = value
        return mask


@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30, model=None, processor=None, device=device, half=False):
    subject_phrases, subject_images = meta.get("subject_phrases"), meta.get("subject_images")
    subject_images = [None] * len(subject_phrases) if subject_images is None else subject_images
    subject_phrases = [None] * len(subject_images) if subject_phrases is None else subject_phrases

    object_phrases, object_images = meta.get("object_phrases"), meta.get("object_images")
    object_images = [None] * len(object_phrases) if object_images is None else object_images
    object_phrases = [None] * len(object_images) if object_phrases is None else object_phrases

    action_phrases, action_images = meta.get("action_phrases"), meta.get("action_images")
    action_images = [None] * len(action_phrases) if action_images is None else action_images
    action_phrases = [None] * len(action_images) if action_phrases is None else action_phrases

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).to(device) if model is None else model
    processor = CLIPProcessor.from_pretrained(version) if processor is None else processor

    subject_boxes = torch.zeros(max_objs, 4)
    object_boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    subject_text_embeddings = torch.zeros(max_objs, 768)
    subject_image_embeddings = torch.zeros(max_objs, 768)
    object_text_embeddings = torch.zeros(max_objs, 768)
    object_image_embeddings = torch.zeros(max_objs, 768)
    action_text_embeddings = torch.zeros(max_objs, 768)
    action_image_embeddings = torch.zeros(max_objs, 768)

    subject_text_features = []
    subject_image_features = []
    object_text_features = []
    object_image_features = []
    action_text_features = []
    action_image_features = []
    for subject_phrase, subject_image, object_phrase, object_image, action_phrase, action_image \
            in zip(subject_phrases, subject_images, object_phrases, object_images, action_phrases, action_images):
        subject_text_features.append(get_clip_feature(model, processor, subject_phrase, is_image=False, device=device))
        subject_image_features.append(get_clip_feature(model, processor, subject_image, is_image=True, device=device))
        object_text_features.append(get_clip_feature(model, processor, object_phrase, is_image=False, device=device))
        object_image_features.append(get_clip_feature(model, processor, object_image, is_image=True, device=device))
        action_text_features.append(get_clip_feature(model, processor, action_phrase, is_image=False, device=device))
        action_image_features.append(get_clip_feature(model, processor, action_image, is_image=True, device=device))

    for idx, (subject_box, object_box,
              subject_text_feature, subject_image_feature,
              object_text_feature, object_image_feature,
              action_text_feature, action_image_feature,) \
            in enumerate(zip(meta['subject_boxes'], meta['object_boxes'],
                             subject_text_features, subject_image_features,
                             object_text_features, object_image_features,
                             action_text_features, action_image_features)):
        if idx >= max_objs:  # no more than max_obj
            break
        subject_boxes[idx] = torch.tensor(subject_box)
        object_boxes[idx] = torch.tensor(object_box)
        masks[idx] = 1
        if subject_text_feature is not None:
            subject_text_embeddings[idx] = subject_text_feature
            object_text_embeddings[idx] = object_text_feature
            action_text_embeddings[idx] = action_text_feature
            text_masks[idx] = 1
        if subject_image_feature is not None:
            subject_image_embeddings[idx] = subject_image_feature
            object_image_embeddings[idx] = object_image_feature
            action_image_embeddings[idx] = action_image_feature
            image_masks[idx] = 1

    if half:
        subject_boxes = subject_boxes.half()
        object_boxes = object_boxes.half()
        masks = masks.half()
        subject_text_embeddings = subject_text_embeddings.half()
        object_text_embeddings = object_text_embeddings.half()
        action_text_embeddings = action_text_embeddings.half()

    out = {
        "subject_boxes": subject_boxes.unsqueeze(0).repeat(batch, 1, 1),
        "object_boxes": object_boxes.unsqueeze(0).repeat(batch, 1, 1),
        "masks": masks.unsqueeze(0).repeat(batch, 1),
        "text_masks": text_masks.unsqueeze(0).repeat(batch, 1) * complete_mask(meta.get("text_mask"), max_objs),
        "image_masks": image_masks.unsqueeze(0).repeat(batch, 1) * complete_mask(meta.get("image_mask"), max_objs),
        "subject_text_embeddings": subject_text_embeddings.unsqueeze(0).repeat(batch, 1, 1),
        "subject_image_embeddings": subject_image_embeddings.unsqueeze(0).repeat(batch, 1, 1),
        "object_text_embeddings": object_text_embeddings.unsqueeze(0).repeat(batch, 1, 1),
        "object_image_embeddings": object_image_embeddings.unsqueeze(0).repeat(batch, 1, 1),
        "action_text_embeddings": action_text_embeddings.unsqueeze(0).repeat(batch, 1, 1),
        "action_image_embeddings": action_image_embeddings.unsqueeze(0).repeat(batch, 1, 1),
    }

    return batch_to_device(out, device)


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize((512, 512))
    return image


@torch.no_grad()
def run(meta, config, starting_noise=None):
    # - - - - - prepare models - - - - - #
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    # - - - - - prepare batch - - - - - #
    batch = prepare_batch(meta, config.batch_size)

    # - - - - - generate prompt context - - - - - #
    context = text_encoder.encode([meta["prompt"]] * config.batch_size)
    uc = text_encoder.encode(config.batch_size * [""])
    if args.negative_prompt is not None:
        uc = text_encoder.encode(config.batch_size * [args.negative_prompt])

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
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
    output_folder = os.path.join(args.folder, meta["save_folder_name"])
    os.makedirs(output_folder, exist_ok=True)

    start = len(os.listdir(output_folder))
    image_ids = list(range(start, start + config.batch_size))
    print(image_ids)
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = str(int(image_id)) + '.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        sample.save(os.path.join(output_folder, img_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="generation_samples", help="root folder for output")

    parser.add_argument("--batch_size", type=int, default=2, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,
                        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
                        help="")
    # parser.add_argument("--negative_prompt", type=str,  default=None, help="")
    args = parser.parse_args()


    clip_text_feature_dict = load_clip_text_cache(device)

    meta_list = [

        dict(
            ckpt="ckpt.pth",
            prompt="a person is feeding a cat",
            subject_phrases=['person'],
            object_phrases=['cat'],
            action_phrases=['feeding'],
            subject_boxes=[[0.0332, 0.1660, 0.3359, 0.7305]],
            object_boxes=[[0.2891, 0.4766, 0.6680, 0.7930]],
            alpha_type=[0.9, 0.0, 0.1],
            save_folder_name="generation_hoi"
        ),
    ]

    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    starting_noise = None
    for meta in meta_list:
        run(meta, args, starting_noise)
