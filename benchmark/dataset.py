from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import random
from torchvision import transforms
import json
import os

class VideoDataset(Dataset):
    def __init__(self, curated_json: str, data_path: str):
        with open(curated_json, 'r') as f:
            self.data = json.load(f)
        self.data_path = data_path
        self.frame_size = (224, 224)
        self.transform = None
    
    def _load_frame(self, video_path: str, frame_idx: int) -> Image.Image:
        """Load and return a single frame as PIL Image."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, self.frame_size)
        
        # Convert to PIL Image
        # frame = Image.fromarray(frame)
        frame = torch.tensor(frame)
            
        return frame

    def load_frames(self, video_id: str, frame_indices: List[int]) -> List[Image.Image]:
        frames = []
        for frame_idx in frame_indices:
            frame = self._load_frame(video_id, frame_idx)
            frames.append(frame)
        return frames
    
    def __getitem__(self, idx: int) -> Tuple[List[Image.Image], Image.Image, List[Image.Image]]:
        sequence = self.data[idx]
        path = os.path.join(self.data_path, sequence['path'])
        input_frames = self.load_frames(path, sequence['frame_indices'][:8])
        target_frame = self.load_frames(path, [sequence['frame_indices'][-1]])[0]
        distractor_frames = self.load_frames(path, sequence['distractor_indices'])
        return input_frames, target_frame, distractor_frames
    
    def __len__(self) -> int:
        return len(self.data)


class VideoDataset_TCV(Dataset):
    def __init__(self, curated_json: str, data_path: str):
        with open(curated_json, 'r') as f:
            self.data = json.load(f)
        self.data_path = data_path
        self.frame_size = (224, 224)
        self.transform = None
    
    def _load_frame(self, video_path: str, frame_idx: int) -> Image.Image:
        """Load and return a single frame as PIL Image."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, self.frame_size)
        
        # Convert to PIL Image
        # frame = Image.fromarray(frame)
        frame = torch.tensor(frame)
            
        return frame

    def load_frames(self, video_id: str, frame_indices: List[int]) -> List[Image.Image]:
        frames = []
        for frame_idx in frame_indices:
            frame = self._load_frame(video_id, frame_idx)
            frames.append(frame)
        return frames
    
    def __getitem__(self, idx: int) -> Tuple[List[Image.Image], Image.Image, List[Image.Image]]:
        sequence = self.data[idx]
        path = os.path.join(self.data_path, sequence['path'])
        all_frames = self.load_frames(path, sequence['frame_indices'][1:9])
        distractor_frame = self.load_frames(path, [sequence['distractor_indices']])
        answer = sequence['answer']
        return all_frames, distractor_frame, answer
    
    def __len__(self) -> int:
        return len(self.data)