import os
from pathlib import Path
from glob import glob
import subprocess
import shutil

dataset_path = "/home/ubuntu/babble-lipsyncs/outputs"

save_dir = "/home/ubuntu/few-shot-vid2vid/dataset/raw/"


data = glob(dataset_path+"/*/")


def _find_filenames(file_dir_path, file_pattern): return list(file_dir_path.glob(file_pattern))


for filePath in data:

    folderName = os.path.basename(os.path.normpath(filePath))

    print(folderName)

    newVideoFolder = save_dir+folderName
    
    original_crop = filePath+'mouth_original'
    wav2lip_crop = filePath+'mouth_cyclic_recon'
    


    

    if not os.path.exists(newVideoFolder):
            os.makedirs(newVideoFolder, 0o777)

    initial_video_frames = newVideoFolder+'/initial'
    ref_video_frames = newVideoFolder+'/reference'

    if not os.path.exists(initial_video_frames):
            os.makedirs(initial_video_frames, 0o777)

    if not os.path.exists(ref_video_frames):
            os.makedirs(ref_video_frames, 0o777)
    
    initial_frames = _find_filenames(Path(original_crop), '*.jpg')
    reference_frames = _find_filenames(Path(wav2lip_crop), '*.jpg')
    
    
    for frame in initial_frames:
        #copy original file to save dir
        shutil.copyfile(frame, initial_video_frames+'/'+os.path.basename(frame))
        
    for frame in reference_frames:
        #copy original file to save dir
        shutil.copyfile(frame, ref_video_frames+'/'+os.path.basename(frame))

    

    

    



