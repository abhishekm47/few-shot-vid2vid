from glob import glob
import os
from pathlib import Path

data_dir = "/home/ubuntu/few-shot-vid2vid/dataset/raw"

data = glob(data_dir+"/*/")

def _find_filenames(file_dir_path, file_pattern): return list(file_dir_path.glob(file_pattern))


for filePath in data:
    first_dir = filePath+'initial'
    second_dir = filePath+'reference'

    files_initial = _find_filenames(Path(first_dir), '*.jpg')
    files_ref = _find_filenames(Path(second_dir), '*.jpg')

    for file_initial in files_initial:
        file_initial = str(file_initial)
        file_ref_ideal = file_initial.replace('/initial/', '/reference/')
        if not os.path.exists(file_ref_ideal):
            print("removing file")
            print(file_initial)
            os.remove(file_initial)

    # if len(files_initial) != len(files_ref):
    #     print(filePath)

    
