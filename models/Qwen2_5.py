from modelscope import AutoModelForCausalLM, AutoTokenizer

class Qwen2_5:
    def __init__(self, parameters = "7B"):
        model_name = "qwen/Qwen2.5-7B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        
    def test_NFS(self, input_frames, all_frames, input_frames_num):
        
        frame_list = ", ".join([f"Frame{i}" for i in range(input_frames_num)])
        input = f"Given a sequence of video frames [{frame_list}]."
              
        input = input + "\n Which one of the following four images is more likely to be the next frame?\n"
        
        input = input + "Your answer should be one of Image0, Image1, Image2, Image3."
        
        messages = [
            {
                "role": "user",
                "content": input
            }
        ]

        # Preparation for inference
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    
    def test_TCV(self, all_frames):
       
        
        frame_list = ", ".join([f"Frame{i}" for i in range(len(all_frames))])
        input = f"Given a sequence of video frames [{frame_list}]."
        
        for i in range(len(all_frames)):
            input = input + f"\nFrame{i}: "
            input = input + f"\n{all_frames[i]}"
        
        input = input + "\n Does any frame look unnatural or not consistent in this video sequence? \n Please answer with yes or no."
        
        messages = [
            {
                "role": "user",
                "content": input
            }
        ]

        # Preparation for inference
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
