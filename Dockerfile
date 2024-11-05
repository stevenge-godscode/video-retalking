# 基础镜像，选择指定版本的 PyTorch 和 CUDA
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# 设置工作目录
WORKDIR /home/workspace/video-retalking-main/

# 复制当前目录的所有文件到容器中
COPY . /home/workspace/video-retalking-main/

# 定义代理参数
ARG HTTP_PROXY
ARG HTTPS_PROXY

# 通过环境变量传递代理信息
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# 设置环境变量以确保 C++ 编译器正确配置
ENV CXX=g++

# 暂时禁用 nvidia 源，并安装必要的软件包
RUN sed -i '/^deb.*nvidia/d' /etc/apt/sources.list.d/* && \
    apt update && \
    apt install -y cmake libgl1-mesa-glx && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# 初始化 bash shell，并创建和激活 conda 环境
RUN conda init bash && \
    conda create -n video_retalking python=3.8 && \
    echo "source activate video_retalking" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && conda activate video_retalking && \
    conda install -c conda-forge ffmpeg && \
    pip install -r requirements.txt"

# 复制 checkpoints 文件夹（假设您本地已经准备好）
COPY checkpoints /home/workspace/video-retalking-main/checkpoints/

# 设置默认启动命令
CMD ["bash"]