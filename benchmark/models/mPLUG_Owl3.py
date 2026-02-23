import torch
import os
from modelscope import AutoConfig, AutoModel
from transformers import AutoTokenizer, AutoProcessor


class mPLUG_Owl3:
    def __init__(self, parameters = "7B"):
        # Set HF_TOKEN environment variable or pass via `huggingface-cli login`
        hf_token = os.getenv("HF_TOKEN")
        
        if parameters == "7B":
            model_path = 'iic/mPLUG-Owl3-7B-241101'
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            print(self.config)  
            self.model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True) 
            self.model.eval()
            
        elif parameters == "2B":
            model_path = 'iic/mPLUG-Owl3-2B-241101'
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, token=hf_token)
            self.model = AutoModel.from_pretrained(
                model_path, 
                attn_implementation='flash_attention_2', 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                token=hf_token
            ).cuda()
            _ = self.model.eval().cuda()
            
        self.device = "cuda"
        self.processor = AutoProcessor.from_pretrained(model_path, token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        
    def test_NFS(self, input_frames, all_frames):
        # Convert tensors to PIL Images if they aren't already
        to_pil = lambda x: Image.fromarray((x.cpu().numpy() * 255).astype('uint8')) if isinstance(x, torch.Tensor) else x
        
        input_frames = [to_pil(frame) for frame in input_frames]
        all_frames = [to_pil(frame) for frame in all_frames]
        
        text = "Give a sequence of video frames <|image|> <|image|> <|image|> <|image|> <|image|> <|image|> <|image|> <|image|>. Which one of the following four images is more likely to be the next frame?\n Image0: <|image|>\n Image1: <|image|>\n Image2: <|image|>\n Image3: <|image|>. \n Your answer should be one of Image0, Image1, Image2, Image3."
        
        messages = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": ""}
        ]
        
        inputs = self.processor(messages, images=input_frames+all_frames, videos=None)
        
        inputs.to(self.device)
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens':100,
            'decode_text':True,
        })

        g = self.model.generate(**inputs)
        print(g)
        return g

