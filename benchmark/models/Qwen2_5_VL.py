from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import torchvision.transforms as T
# default: Load the model on the available device(s)


class Qwen2_5_VL:
    def __init__(self, parameters = "7B"):
        
        if parameters == "7B":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            
        elif parameters == "3B":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        
    def test_NFS(self, input_frames, all_frames, input_frames_num):
        # Convert tensors to PIL Images if they aren't already
        to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
        
        input_frames = [to_pil(frame) for frame in input_frames]
        all_frames = [to_pil(frame) for frame in all_frames]
        
        frame_list = ", ".join([f"Frame{i}" for i in range(input_frames_num)])
        input_lists = [{"type": "text", "text": f"Given a sequence of video frames [{frame_list}]."}]
        
        for i in range(input_frames_num):
            input_lists.append({"type": "text", "text": f"\nFrame{i}: "})
            input_lists.append({"type": "image", "image": input_frames[i + len(all_frames) - input_frames_num]})
        
        input_lists.append({"type": "text", "text": "\n Which one of the following four images is more likely to be the next frame?\n"})
        for i in range(len(all_frames)):
            input_lists.append({"type": "text", "text": f"Image{i}: "})
            input_lists.append({"type": "image", "image": all_frames[i]})
        
        input_lists.append({"type": "text", "text": "Your answer should be one of Image0, Image1, Image2, Image3."})
        messages = [
            {
                "role": "user",
                "content": input_lists
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)  # Move inputs to same device as model

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
    
    
    def test_TCV(self, all_frames):
       
        # Convert tensors to PIL Images if they aren't already
        to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
        
        all_frames = [to_pil(frame) for frame in all_frames]
        
        frame_list = ", ".join([f"Frame{i}" for i in range(len(all_frames))])
        input_lists = [{"type": "text", "text": f"Given a sequence of video frames [{frame_list}]."}]
        
        for i in range(len(all_frames)):
            input_lists.append({"type": "text", "text": f"\nFrame{i}: "})
            input_lists.append({"type": "image", "image": all_frames[i]})
        
        input_lists.append({"type": "text", "text": "\n Does any frame look unnatural or not consistent in this video sequence? \n Please answer with yes or no."})
        messages = [
            {
                "role": "user",
                "content": input_lists
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)  # Move inputs to same device as model

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
