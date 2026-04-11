import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

class llava_interleave:
    def __init__(self, parameters = "7B"):
        
        if parameters == "7B":
            model_id = "llava-hf/llava-interleave-qwen-7b-hf"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
            ).to("cuda")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.processor.patch_size = 14
            self.processor.vision_feature_select_strategy = "full"

    def test_NFS(self, input_frames, all_frames, input_frames_num):
        # Convert tensors to PIL Images if they aren't already
        to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
        
        input_frames = [to_pil(frame) for frame in input_frames]
        all_frames = [to_pil(frame) for frame in all_frames]
        
        frame_list = ", ".join([f"Frame{i}" for i in range(input_frames_num)])
        input_lists = [{"type": "text", "text": f"Give a sequence of video frames [{frame_list}]."}]
        
        for i in range(input_frames_num):
            input_lists.append({"type": "text", "text": f"\nFrame{i}: "})
            input_lists.append({"type": "image", "image": input_frames[i + len(all_frames) - input_frames_num]})
        
        input_lists.append({"type": "text", "text": "\n Which one of the following four images is more likely to be the next frame?\n"})
        for i in range(len(all_frames)):
            input_lists.append({"type": "text", "text": f"Image{i}: "})
            input_lists.append({"type": "image", "image": all_frames[i]})
        
        input_lists.append({"type": "text", "text": "Your answer should be one of Image0, Image1, Image2, Image3."})
        conversation = [
            {
                "role": "user",
                "content": input_lists
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        # Move inputs to the same device as the model
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        output = self.model.generate(**inputs, max_new_tokens=50)
        print(self.processor.decode(output[0][2:], skip_special_tokens=True))   
        return self.processor.decode(output[0][2:], skip_special_tokens=True)


