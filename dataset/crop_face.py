import os
from pathlib import Path
from glob import glob
import subprocess
import cv2
import sys
import torch 
from tqdm import tqdm
import numpy as np
import platform

sys.path.append('/home/ubuntu/making_with_ml/ai_dubs')

dataset_path = "/home/ubuntu/few-shot-vid2vid/dataset/pre_processed"

import Wav2Lip.face_detection as face_detection

face_det_batch_size = 16
resize_factor = 1
rotate = False
bbox_pads = [0, 10, 0, 0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nosmooth = False 

final_img_size = 96



def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def crop_face_from_frame_with_bbox(frame, bbox):

    increase_area = 0.10
    top, bot, left, right  = bbox
    width = right - left
    height = bot - top
    frame_shape = frame.shape

    
    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
    
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)
    h, w = bot - top, right - left
    
    top2, bot2, left2, right2 = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    crop_img = frame[top2:bot2, left2:right2]
    
    top_border = abs(top2 - top)
    bot_border = abs(bot2 - bot)
    left_border = abs(left2 - left)
    right_border = abs(right2 - right)
    
    crop_img = cv2.copyMakeBorder(crop_img, top=top_border, bottom=bot_border, left=left_border, right=right_border, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    crop_img = cv2.resize(crop_img, (256, 256))
    
    #crop_img = cv2.flip( crop_img, 0 ) 
    crop_img = cv2.flip( crop_img, 1 )
    
    return crop_img




def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = bbox_pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('/home/ubuntu/making_with_ml/ai_dubs/Wav2Lip/temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image, (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 




def process_output():
    data = glob(dataset_path+"/*/")
    for file in data:
        print('processing: {}'.format(file))
        crop_file = Path(file+'crop.mp4')
        if crop_file.is_file():
            continue
        else:
            process_video(file)






def process_video(file):
    origin_file = file+'original.mp4'
    save_file = file+'crop.mp4'
    temp_file = file+'temp.avi'
    video_stream = cv2.VideoCapture(origin_file)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        
        full_frames.append(frame)

    bboxes_face = face_detect(full_frames)
    
    crop_imgs = []

    videox = cv2.VideoWriter(temp_file, 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (256,256))

    for i, (frame, bbox) in enumerate(bboxes_face):
        this_crop = crop_face_from_frame_with_bbox(frame, bbox)
        crop_imgs.append(this_crop)
        videox.write(this_crop)
    
    videox.release()
    
    subprocess.call(['ffmpeg', '-i', temp_file, '-i', origin_file, '-map', '0:v:0', '-map', '1:a:0', '-shortest', save_file])


if __name__ == "__main__":
    process_output()
        


