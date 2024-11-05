 docker run --gpus all --name retalking --ipc=host -it     -v /home/workspace/video-retalking-main:/workspace/video-retalking-main     -v /home/workspace/file-share:/workspace/file-share -v ~/.cache:/root/.cache    pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel /bin/bash

cd /home/workspace/video-retalking-main/
#conda create -n video_retalking python=3.8

conda activate video_retalking

conda install ffmpeg
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
apt update
apt install -y cmake
# pip install dlib
pip install -r requirements.txt

export CXX=g++
#export HTTP_PROXY=http://localhost:3128
#export HTTPS_PROXY=http://localhost:3128

apt update
apt install -y libgl1-mesa-glx
export TORCH_CUDA_ARCH_LIST="8.6"

nohup python3 inference.py --face examples/face/1.mp4 --audio examples/audio/1.wav --outfile results/1_1.mp4 &

nohup python3 inference.py --face examples/face/G2.mp4 --audio examples/audio/EN1.wav --outfile results/G2.mp4 &


tail -n 100 -f nohup
#export CUDA_VISIBLE_DEVICES=""


docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel