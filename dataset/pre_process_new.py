import sys

sys.path.append('/home/ubuntu/making_with_ml/ai_dubs')

from dubber import dub
from Wav2Lip.Wav2Lip_frame import wav2lip_lip_frame

import shutil
import subprocess
import os
from pathlib import Path
from glob import glob

output_size  = 512

dataset_path = "/home/ubuntu/few-shot-vid2vid/dataset/pre_processed"

save_dir = "/home/ubuntu/few-shot-vid2vid/dataset/new_crop"

dubSrc = False

sourcLang = 'en'

targetLang = 'hi'

storageBucket = 'copycat_ai_dubs'

voice_obj_1 = 'hi-IN-Wavenet-B'

voice_obj_2 = 'en-IN-Wavenet-B'

def _find_filenames(file_dir_path, file_pattern): return list(file_dir_path.glob(file_pattern))




def pre_process_videos():
    #path of all files
    data = glob(dataset_path+"/*/")
    dataset =  [filename for filename in data]

    print('loaded {} files ...'.format(len(dataset)))

    files_count = len(dataset)
    processed = 1

    for video in dataset:
        file_name = video.split('/')[6]
        
        print("processing '{}' {}/{}".format(file_name, processed, files_count))
        
        processed = processed+1


        file_save_dir = save_dir+'/'+file_name

        if not os.path.exists(file_save_dir):
            os.makedirs(file_save_dir, 0o777)
        else:
            continue

        original_file = video+'original.mp4'

        #copy original file to save dir
        shutil.copyfile(original_file, file_save_dir+'/original.mp4')

        first_run_dir = file_save_dir+'/initial'

        if not os.path.exists(first_run_dir):
            os.makedirs(first_run_dir, 0o777)

        
        #run dubbing
        dub(original_file, first_run_dir, sourcLang, [targetLang], storageBucket, [], dubSrc, 1, {targetLang:voice_obj_1})
        
        #create folder for saving frames 
        original_folder = first_run_dir+'/original'
        if not os.path.exists(original_folder):
            os.makedirs(original_folder, 0o777)
        
        #create folder for saving lipsync outputs
        lipSync_folder = first_run_dir+'/Wav2Lip'
        if not os.path.exists(lipSync_folder):
            os.makedirs(lipSync_folder, 0o777)

        #create folder for saving lipsync outputs
        crop_folder = first_run_dir+'/Wav2LipCrop'
        if not os.path.exists(crop_folder):
            os.makedirs(crop_folder, 0o777)

        #create folder for saving original crop
        original_crop_folder = first_run_dir+'/OriginalCrop'
        if not os.path.exists(original_crop_folder):
            os.makedirs(original_crop_folder, 0o777)
            

        #get all dubbed files
        dubbed_videos_dir = first_run_dir+'/dubbedVideos'
        dubbed_video_filenames = _find_filenames(Path(dubbed_videos_dir), '*.mp4')
        dubbed_video_filenames =  [str(filename) for filename in dubbed_video_filenames]


        for ai_dubs in dubbed_video_filenames:
            frames_save_string = original_folder + "/image-%07d.jpg"
            
            subprocess.call(['ffmpeg',  '-noautorotate', '-i', ai_dubs, '-f', 'image2', '-q:v', '1', frames_save_string])
            
            wav2lip_lip_frame(original_folder, ai_dubs, lipSync_folder, crop_folder, original_crop_folder)


        #second stage network

        second_run_dir = file_save_dir+'/final'

        if not os.path.exists(second_run_dir):
            os.makedirs(second_run_dir, 0o777)

        

        #create folder for saving lipsync outputs
        second_lipSync_folder = second_run_dir+'/Wav2Lip'
            
        if not os.path.exists(second_lipSync_folder):
            os.makedirs(second_lipSync_folder, 0o777)

        #create folder for saving lipsync outputs
        second_lipSyncCrop_folder = second_run_dir+'/Wav2LipCrop'
            
        if not os.path.exists(second_lipSyncCrop_folder):
            os.makedirs(second_lipSyncCrop_folder, 0o777)

        second_OriginalCrop_folder = second_run_dir+'/OriginalCrop'
            
        if not os.path.exists(second_OriginalCrop_folder):
            os.makedirs(second_OriginalCrop_folder, 0o777)
            
            
        wav2lip_lip_frame(lipSync_folder, original_file, second_lipSync_folder, second_lipSyncCrop_folder, second_OriginalCrop_folder)


        










if __name__ == "__main__":
    pre_process_videos()



        
            



        



    