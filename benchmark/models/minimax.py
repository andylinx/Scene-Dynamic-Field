import base64
from openai import OpenAI
from PIL import Image
import io
class MiniMax:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.minimaxi.chat/v1",
        )
        self.system_message = {
            "role": "system",
            "content": "MM Intelligent Assistant is a large language model that is self-developed by MiniMax and does not call the interface of other products."
        }

    def process_image(self, image_path=None, image_url=None):
        """Process either a local image or image URL"""
        if image_path:
            image_file = image_path
            
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image_file.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            data = base64.b64encode(img_byte_arr).decode('utf-8')
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{data}"
                }
            }
        elif image_url:
            return {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        return None

    def generate_response(self, text, images=None):
        """Generate response for text and optional images"""
        content = [{"type": "text", "text": text}]
        
        if images:
            for img in images:
                if isinstance(img, str):
                    # Treat as URL if string
                    image_data = self.process_image(image_url=img)
                else:
                    # Treat as file path if not string
                    image_data = self.process_image(image_path=img)
                if image_data:
                    content.append(image_data)

        messages = [
            self.system_message,
            {
                "role": "user",
                "name": "user",
                "content": content
            }
        ]

        completion = self.client.chat.completions.create(
            model="MiniMax-Text-01",
            messages=messages,
            max_tokens=4096,
        )
        print(completion)
        return completion.choices[0].message

    def test_NFS(self, input_frames, all_frames):
        """
        Analyze a sequence of frames and predict the next frame from given options.
        
        Args:
            input_frames: List of PIL Image objects representing the sequence
            all_frames: List of PIL Image objects representing the possible next frames
        """
        # Create the prompt with input frames
        frame_descriptions = [f"Frame{i}: " for i in range(len(input_frames))]
        frame_sequence = ", ".join(frame_descriptions)
        
        prompt = f"Give a sequence of video frames [{frame_sequence}]. You should carefully pay attention to the dynamics of fluid in the video, and determine which one of the following four images is more likely to be the next frame?\n"
        
        # Add the candidate frames to the prompt
        for i in range(len(all_frames)):
            prompt += f"Image{i}: \n"
        
        prompt += "Your answer should be one of Image0, Image1, Image2, Image3, and explain your reasoning."

        # Process all images (both input frames and candidate frames)
        all_images = input_frames + all_frames
        
        # Generate response using the existing method
        response = self.generate_response(prompt, images=all_images)
        
        return response.content