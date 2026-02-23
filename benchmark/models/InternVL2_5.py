import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
from lmdeploy import pipeline, TurbomindEngineConfig, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from typing import List, Tuple, Dict
import numpy as np
import os
import warnings
import json
import cv2

warnings.filterwarnings("ignore")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    images = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    # images = transform(image)
    images = dynamic_preprocess(images, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
    return pixel_values

peft_module_dict = {
    '1B': ['mlp1.1', 'mlp1.3'],
    '2B': ['mlp1.1', 'mlp1.3', 'qkv', 'fc1', 'fc2', 'proj', 'wqkv', 'wo', 'w1', 'w2', 'w3', 'output'],
    '4B': ['mlp1.1', 'mlp1.3', 'qkv', 'proj', 'fc1', 'fc2', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    '8B': ['mlp1.1', 'mlp1.3', 'qkv', 'proj', 'fc1', 'fc2', 'wqkv', 'wo', 'w1', 'w2', 'w3', 'output'],
}
max_l_dict = {
    '1B': 1357,
    '2B': 1417,
    '4B': 1405,
}

class InternVL2_5(nn.Module):
    
    def __init__(self, parameters = "8B"):
        super(InternVL2_5, self).__init__()
        path = f'OpenGVLab/InternVL2_5-{parameters}'
        self.pipe = pipeline(path, backend_config=PytorchEngineConfig(session_len=8192))
     
    def test_NFS(self, input_frames, all_frames, input_frames_num, mode='test'):
        if mode == 'test':
            pixel_values = []
            input_lists = []
            
            to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
            input_frames = [to_pil(frame) for frame in input_frames]
            all_frames = [to_pil(frame) for frame in all_frames]
            
            input_lists.append("Given a sequence of video frames:\n")
            
            for i in range(input_frames_num):
                input_lists.append(f'Frame-{i}: {IMAGE_TOKEN}\n')
                pixel_values.append(input_frames[i + len(input_frames) - input_frames_num])
            
            input_lists.append("Which one of the following four images is more likely to be the next frame?\n")
            
            for i in range(len(all_frames)):
                input_lists.append(f"Image{i}: {IMAGE_TOKEN}\n")
                pixel_values.append(all_frames[i])

            ques = ''.join(input_lists)
            
            ques = ques + "Your answer should be one of Image0, Image1, Image2, Image3."
            response = self.pipe((ques, pixel_values))
            # print(response)
            # print(ques)
            # print(len(pixel_values))
            return response.text
    
    def test_TCV(self, all_frames, mode='test'):
        
        if mode == 'test':
        
            pixel_values = []
            input_lists = []
            
            to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
            all_frames = [to_pil(frame) for frame in all_frames]
            
            # for i in range(len(all_frames)):
            #     print(type(all_frames[i]))
            
            input_lists.append("Given a sequence of video frames:\n")
            
            for i in range(len(all_frames)):
                input_lists.append(f'Frame-{i}: {IMAGE_TOKEN}\n')
                pixel_values.append(all_frames[i])
                
            ques = ''.join(input_lists)
            ques = ques + "\n Does any frame look unnatural or not consistent in this video sequence? \n Please answer with yes or no."
            response = self.pipe((ques, pixel_values))
            
            return response.text
        
            