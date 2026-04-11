import requests
import torch
from PIL import Image
import base64

def image_to_base64(image):
    
    image.save("image.png")
    with open("image.png", "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

import os

def get_environment_token():
    return os.getenv("siliconflow_token")

class api:
    def __init__(self, model_name):
        if model_name == "Qwen2_VL_api":
            self.model_name = "Qwen/Qwen2-VL-72B-Instruct"
        elif model_name == "InternVL2_5_api":
            self.model_name = "OpenGVLab/InternVL2-26B"
            
    def test_NFS(self, input_frames, all_frames, input_frames_num):
        # Convert tensors to PIL Images if they aren't already
        to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
        
        input_frames = [to_pil(frame) for frame in input_frames]
        all_frames = [to_pil(frame) for frame in all_frames]
        
        frame_list = ", ".join([f"Frame{i}" for i in range(input_frames_num)])
        input_lists = [{"type": "text", "text": f"Given a sequence of video frames [{frame_list}]."}]
        
        for i in range(input_frames_num):
            input_lists.append({"type": "text", "text": f"\nFrame{i}: "})
            base64_image = image_to_base64(input_frames[i + len(all_frames) - input_frames_num])
            input_lists.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        
        input_lists.append({"type": "text", "text": "\n Which one of the following four images is more likely to be the next frame?\n"})
        for i in range(len(all_frames)):
            input_lists.append({"type": "text", "text": f"Image{i}: "})
            base64_image = image_to_base64(all_frames[i])
            input_lists.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        
        input_lists.append({"type": "text", "text": "Your answer should be one of Image0, Image1, Image2, Image3."})
       
        url = "https://api.siliconflow.cn/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": input_lists
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "stop": ["null"],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }
        headers = {
            "Authorization": "Bearer " + get_environment_token(),
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        print(response)
        print(response.text)
        
        return response.text
    
    
    def test_TCV(self, all_frames):
       
        # Convert tensors to PIL Images if they aren't already
        to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
        
        all_frames = [to_pil(frame) for frame in all_frames]
        
        frame_list = ", ".join([f"Frame{i}" for i in range(len(all_frames))])
        input_lists = [{"type": "text", "text": f"Given a sequence of video frames [{frame_list}]."}]
        
        for i in range(len(all_frames)):
            input_lists.append({"type": "text", "text": f"\nFrame{i}: "})
            base64_image = image_to_base64(all_frames[i])
            input_lists.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        
        input_lists.append({"type": "text", "text": "\n Does any frame look unnatural or not consistent in this video sequence? \n Please answer with yes or no."})

        url = "https://api.siliconflow.cn/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": input_lists
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "stop": ["null"],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }
        headers = {
            "Authorization": "Bearer " + get_environment_token(),
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        
        return response.text
