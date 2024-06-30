import torch.nn as nn

class VideoDecoder(nn.Module):
    def __init__(self, hidden_dim, num_frames):
        super(VideoDecoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim * num_frames * 4)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

    def forward(self, text_features):
        x = self.fc(text_features).view(-1, self.hidden_dim, self.num_frames, 4)
        video_output = self.deconv_layers(x)
        return video_output