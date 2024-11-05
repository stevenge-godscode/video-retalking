import os
import cv2
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from itertools import cycle
from torch.multiprocessing import Pool, set_start_method

# 定义打印CUDA信息的函数
def print_cuda_info():
    print("CUDA Information:")
    if torch.cuda.is_available():
        print("  - CUDA is available")
        print("  - Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"    - Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
    else:
        print("  - CUDA is not available, using CPU instead")

class KeypointExtractor():
    def __init__(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)

    def extract_keypoint(self, images, name=None, info=True):
        # 用于提取多张图片的关键点
        if isinstance(images, list):
            keypoints = []
            i_range = tqdm(images, desc='Landmark Detection:') if info else images

            for image in i_range:
                # 使用辅助方法处理单张图像
                current_kp = self._extract_single_keypoint(image)
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0] + '.txt', keypoints.reshape(-1))
            return keypoints
        else:
            # 处理单张图像的关键点提取
            return self._extract_single_keypoint(images, name)

    def _extract_single_keypoint(self, image, name=None):
        # 辅助方法：用于单张图像的关键点提取
        while True:
            try:
                keypoints = self.detector.get_landmarks_from_image(np.array(image))[0]
                break
            except RuntimeError as e:
                if str(e).startswith('CUDA'):
                    print("Warning: out of memory, sleep for 1s")
                    time.sleep(1)
                else:
                    print(e)
                    break    
            except TypeError:
                print("No face detected in this image")
                keypoints = -1. * np.ones([68, 2])  # Default shape for no face
                break
        if name:
            np.savetxt(os.path.splitext(name)[0] + '.txt', keypoints.reshape(-1))
        return keypoints
    
def read_video(filename):
    print(f"Reading video: {filename}")
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        else:
            break
    cap.release()
    print(f"Total frames extracted: {len(frames)}")
    return frames

def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    print_cuda_info()  # 打印每次运行的CUDA信息
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split('/')[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(images, name=os.path.join(opt.output_dir, name[-2], name[-1]))

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='Folder with input files')
    parser.add_argument('--output_dir', type=str, help='Folder for output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()
    filenames = []
    VIDEO_EXTENSIONS = {'mp4', 'MP4'}

    for ext in VIDEO_EXTENSIONS:
        filenames += glob.glob(f'{opt.input_dir}/*.{ext}')
    filenames = sorted(filenames)

    print('Total number of videos:', len(filenames))
    print_cuda_info()  # 程序启动时打印CUDA信息

    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = cycle(opt.device_ids.split(","))
    
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        pass