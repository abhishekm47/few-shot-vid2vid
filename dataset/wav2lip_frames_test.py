import sys
import os
from pathlib import Path
from glob import glob
import subprocess

sys.path.append('/home/ubuntu/making_with_ml/ai_dubs')

from Wav2Lip.Wav2Lip_frame import wav2lip_lip_frame


file_path = "/home/ubuntu/few-shot-vid2vid/dataset/tiktokVid/Snaptik_6672751607819537669_olly-ince.mp4"

dir_name = os.path.basename(file_path).split('.')[0]

if not os.path.exists(dir_name):
            os.makedirs(dir_name, 0o777)
            
save_dir = dir_name+'/original'

if not os.path.exists(save_dir):
            os.makedirs(save_dir, 0o777)

frames_save_string = save_dir + "/image-%07d.jpg"

subprocess.call(['ffmpeg',  '-noautorotate', '-i', file_path, '-f', 'image2', '-q:v', '1', frames_save_string])

results = wav2lip_lip_frame(save_dir, file_path, dir_name+'/Wav2Lip_large', dir_name+'/Wav2Lip_crop', dir_name+'/crop_original')

print(results)
