import sys
import os
# Allow importing model wrappers from the models/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

from dataset import VideoDataset, VideoDataset_TCV
import argparse
import re
import random
from logger import BenchmarkLogger
from tqdm import tqdm
from minimax import MiniMax

def parse_args():
    parser = argparse.ArgumentParser(description='Video Model Benchmark')
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['Qwen2_5', 'Qwen2_VL', "Qwen2_VL_api", 'Qwen2_5_VL', 'mPLUG_Owl3', 'InternVL2_5', 'InternVL2_5_api','InternVideo2_5', 'VideoChat2', 'MiniMax', 'llava-interleave'],
        required=True,
        help='Model to use for benchmarking'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['NFS', 'TCV'],
        required=True,
        help='Type of benchmark to run'
    )
    
    
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='JSON file to use'
    )
    
    parser.add_argument(
        '--input_frames',
        type=int,
        default=8,
        required=False,
        help='Number of input frames'
    )
    
    parser.add_argument(
        '--parameters',
        type=str,
        choices=['1B', '2B', '3B', '4B', '7B', '8B', '72B'],
        required=False,
        default="7B",
        help='Parameters size'
    )
    
    return parser.parse_args()

def check_answer(response, target, type):
    if type == "NFS":
        pattern = "Image" + str(target)
        match = re.search(pattern, response)
        return match is not None

    elif type == "TCV":
        response = response.lower()
        match = re.search(target, response)
        return match is not None

def TCV_benchmark(args, model):
    
    data_file = args.json_file
    dataset = VideoDataset_TCV(data_file, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "videos"))
    logger = BenchmarkLogger(args.model, "TCV", args.input_frames, args.json_file, args.parameters)
    
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        all_frames, distractor_frame, answer = dataset[i]
        
        all_frames = all_frames[-args.input_frames:]
        if answer == "no":
            random_distractor = random.randint(1, len(all_frames) - 1)
            all_frames[random_distractor] = distractor_frame[0]
        
        response = model.test_TCV(all_frames)
        correct = check_answer(response, answer, "TCV")
        logger.log_sample(i, response, answer, correct, response)
        
        with open("benchmark_log.txt", "a") as log_file:
            log_file.write(f"Sample {i} of {len(dataset)}:\n")
            log_file.write(f"Response: {response}\n")
            log_file.write(f"Target ID: {answer}\n")
            
    
    logger.print_summary()
    logger.save()


def NFS_benchmark(args, model):
    data_file = args.json_file
    dataset = VideoDataset(data_file, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "videos"))
    logger = BenchmarkLogger(args.model, "NFS", args.input_frames, args.json_file, args.parameters)
    
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        
        input_frames, target_frame, distractor_frames = dataset[i]
         # Create list of frames with known target position
        all_frames = [target_frame] + distractor_frames
        indices = list(range(len(all_frames)))  # [0,1,2,3]
        
        # Shuffle both lists using the same random state
        combined = list(zip(all_frames, indices))
        random.shuffle(combined)
        all_frames, shuffled_indices = zip(*combined)
        
        # Find where our target (originally at index 0) went
        target_id = shuffled_indices.index(0)

        response = model.test_NFS(input_frames, all_frames, args.input_frames)
        correct = check_answer(response, target_id, "NFS")
        logger.log_sample(i, response, str(target_id), correct, response)
        
        with open("benchmark_log.txt", "a") as log_file:
            log_file.write(f"Sample {i} of {len(dataset)}:\n")
            log_file.write(f"Response: {response}\n")
            log_file.write(f"Target ID: {target_id}\n")
            
    logger.print_summary()
    logger.save()

if __name__ == "__main__":
    
    args = parse_args()
    
    if args.model == "Qwen2_VL":
        from Qwen2_VL import Qwen2_VL
        model = Qwen2_VL(args.parameters)
    elif args.model == "Qwen2_5":
        from Qwen2_5 import Qwen2_5
        model = Qwen2_5(args.parameters)
    elif args.model == "Qwen2_5_VL":
        from Qwen2_5_VL import Qwen2_5_VL
        model = Qwen2_5_VL(args.parameters)
    elif args.model == "Qwen2_VL_api":
        from api import api
        model = api(args.model)
    elif args.model == "llava-interleave":
        from llava_interleave import llava_interleave
        model = llava_interleave()
    elif args.model == "mPLUG_Owl3":
        from mPLUG_Owl3 import mPLUG_Owl3
        model = mPLUG_Owl3()
    elif args.model == "InternVL2_5":
        from InternVL2_5 import InternVL2_5
        model = InternVL2_5(args.parameters)
    elif args.model == "InternVL2_5_api":
        from api import api
        model = api(args.model)
    elif args.model == "InternVideo2_5":
        from InternVideo2_5 import InternVideo2_5
        model = InternVideo2_5()
    elif args.model == "VideoChat2":
        from VideoChat2 import VideoChat2
        model = VideoChat2()    
    elif args.model == "MiniMax":
        model = MiniMax("your api key here")

    if args.type == "NFS":
        NFS_benchmark(args, model)
    elif args.type == "TCV":
        TCV_benchmark(args, model)
