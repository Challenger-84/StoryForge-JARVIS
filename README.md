# StoryForge-JARVIS

Welcome to the StoryForge project! This project showcases a cutting-edge Text-to-Video (T2V) model that transforms textual descriptions into dynamic videos. We have utilized two powerful models in our code:

1. **Stable Diffusion** for Text-to-Image (T2I) generation.
2. **Stable Video diffusion** for Image-to-Video (I2V) generation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Models](#models)
  - [Stable Diffusion](#stable-diffusion)
  - [Stable Video diffusion](#stable-video-diffusion)
- [Contributors](#contributors)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The StoryForge project aims to bridge the gap between text and video generation by leveraging state-of-the-art models. By inputting a textual description, our pipeline generates a sequence of images using Stable Diffusion and then converts these images into a cohesive video using SVD.

## Installation

To get started, follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Challenger-84/StoryForge-JARVIS.git
   cd StoryForge-JARVIS
   

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required dependencies:
   ```bash
   pip install -r requirement.txt

 4. Model weights are available on Google drive:
https://drive.google.com/drive/folders/1k3vvyncTzvh47j39fzwbioZPZRuQl8w6?usp=sharing

## Models

# Stable diffusion
Stable Diffusion is a deep learning model released by Stability.ai that generates high-quality images from textual descriptions. It leverages a diffusion process to iteratively improve image quality and coherence.

# Stable Video diffusion
Stable Video Diffusion (SVD) Image-to-Video is a diffusion model that takes in a still image as a conditioning frame, and generates a video from it.

## Contributors
Parikshit Gehlaut   
R Eshwar

## Contributing
We welcome contributions to the our project! To contribute, please follow these steps:

1. Fork the repository
2. create a new branch
3. Make and commit your changes
4. Push to the branch
5. Open a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.

