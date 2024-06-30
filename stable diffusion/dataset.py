import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames_per_clip=100):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames_per_clip = max_frames_per_clip
        self.clips, self.labels = self.load_clips_and_labels()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        frames = self.read_clip_frames(clip_path)
        
        frames = self.pad_or_trim_frames(frames, self.max_frames_per_clip)

        if self.transform:
            transformed_frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(transformed_frames)
        
        label = self.labels[idx]
        
        return frames, label
    
    def load_clips_and_labels(self):
        clips = []
        labels = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.avi') or filename.endswith('.mp4'):
                    clips.append(os.path.join(dirpath, filename))
                    labels.append(os.path.basename(dirpath))
        return clips, labels
    
    def read_clip_frames(self, clip_path):
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)
        cap.release()
        return frames
    
    def pad_or_trim_frames(self, frames, max_frames):
        num_frames = len(frames)
        if num_frames < max_frames:
            frames.extend([frames[-1]] * (max_frames - num_frames))
        elif num_frames > max_frames:
            frames = frames[:max_frames]
        return frames


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 171)), 
        transforms.CenterCrop(112),    
        transforms.ToTensor()          
    ])

    root_dir = './data/UCF101/UCF-101'
    dataset = UCF101Dataset(root_dir, transform=transform, max_frames_per_clip=100)

    frames, label = dataset[0]
    print(frames.shape, label)
