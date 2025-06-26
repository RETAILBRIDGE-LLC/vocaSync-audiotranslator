#!/bin/bash

# Exit on any error
#set -e

# Load Conda (adjust path if needed)
source ~/anaconda3/etc/profile.d/conda.sh

# Go to the project directory
cd /home/ec2-user/SageMaker/vocaSync-audiotranslator/

# Create Conda environment
if ! conda info --envs | grep -q 'tts-hs-hifigan'; then
    conda env create -f environment.yml
else
    echo "🔁 Conda environment 'tts-hs-hifigan' already exists. Skipping creation."
fi


# Initialize Conda in bash
conda init bash

# Activate the environment
conda activate tts-hs-hifigan

# Clone HiFi-GAN repo
#git clone https://github.com/jik876/hifi-gan.git   
cd hifi-gan

# Install HiFi-GAN requirements
pip install -r requirements.txt
pip install --upgrade librosa

# Return to project directory
cd /home/ec2-user/SageMaker/vocaSync-audiotranslator/

# Install PyTorch and related packages (CPU-only)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install project Python dependencies
pip install -r requirements.txt

# Conda dependencies for audio and compatibility
conda install -c conda-forge libsndfile -y
conda install -c conda-forge llvmlite=0.36.0 numba=0.53.1 -y

# Run a test inference
python inference.py \
  --sample_text "Hi team. We are now able to generate speech using Fast speech 2" \
  --language english \
  --gender male \
  --alpha 1 \
  --output_file male_english_output.wav

echo "✅ Setup complete and inference ran successfully!"

