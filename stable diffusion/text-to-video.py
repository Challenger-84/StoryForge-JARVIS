import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import cv2
import numpy as np

from textencoder import TextEncoder
from videodecoder import VideoDecoder
from dataset import UCF101Dataset

class TextToVideoModel(nn.Module):
    def __init__(self, text_encoder, video_decoder):
        super(TextToVideoModel, self).__init__()
        self.text_encoder = text_encoder
        self.video_decoder = video_decoder
        
    def forward(self, text):
        text_features = self.text_encoder(text)
        video_output = self.video_decoder(text_features)
        return video_output
    
def main():
    hidden_dim = 768 
    num_frames = 16  

    text_encoder = TextEncoder().cuda()
    video_decoder = VideoDecoder(hidden_dim, num_frames).cuda()
    model = TextToVideoModel(text_encoder, video_decoder).cuda() 
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    transform = transforms.Compose([
        transforms.Resize((128, 171)),  
        transforms.CenterCrop(112),   
        transforms.ToTensor()         
    ])

    root_dir = './data/UCF101/UCF-101'
    dataset = UCF101Dataset(root_dir, transform=transform)

    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Training parameters
    num_epochs = 100
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for frames, labels in train_loader:
            frames = frames.cuda()
            labels = list(labels) 
            
            optimizer.zero_grad()
            outputs = model(labels) 

            batch_size, num_frames_gt, channels, height, width = frames.size()
            frames_resized = frames.view(batch_size, num_frames_gt, channels, height, width)

            
            num_frames_outputs = outputs.size(1)
            outputs = outputs[:, :num_frames_outputs, ...] 

            print(outputs.size())
            print(frames_resized.size())

            # Calculate MSE loss
            loss = criterion(outputs, frames_resized)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * frames.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, text, labels in val_loader:
                frames = frames.cuda()
                text = list(text)  
                outputs = model(text)
                
               
                batch_size_val, num_frames_val, channels_val, height_val, width_val = frames.size()
                frames_reshaped_val = frames.view(batch_size_val * num_frames_val, channels_val, height_val, width_val)
                frames_resized_val = F.interpolate(frames_reshaped_val, size=(128, 32), mode='bilinear', align_corners=False)
                frames_resized_val = frames_resized_val.view(batch_size_val, num_frames_val, channels_val, 128, 32)

                num_frames_outputs_val = outputs.size(1)
                frames_resized_val = frames_resized_val[:, :num_frames_outputs_val, ...]

                loss = criterion(outputs, frames_resized_val)
                val_loss += loss.item() * frames.size(0)

        val_loss /= len(val_dataset)
        print(f'Validation Loss: {val_loss:.4f}')
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print("Early stopping")
            break

    print("Training complete.")

    def generate_video(model, text_input, output_file='generated_video.mp4', frame_size=(112, 112), fps=25):
        model.eval()
        with torch.no_grad():
            generated_video = model([text_input])
        generated_video = generated_video.squeeze().permute(0, 2, 3, 1).numpy() 

        generated_video = (generated_video * 255).astype(np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

        for frame in generated_video:
            out.write(frame)

        out.release()

    text_encoder = TextEncoder().cuda()
    video_decoder = VideoDecoder(hidden_dim=768, num_frames=16).cuda()
    model = TextToVideoModel(text_encoder, video_decoder).cuda()
    model.load_state_dict(torch.load('best_model.pth'))

    text_input = "a rabbit running"
    generate_video(model, text_input, output_file='generated_video.mp4')


if __name__ == "__main__":
    main()
