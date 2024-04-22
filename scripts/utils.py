from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import cv2
import pandas as pd
import torch 
import h5py
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchvision import transforms
from skimage import io
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import cv2
import pandas as pd
import torch 
import h5py
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchvision import transforms
from skimage import io

class VideoLoader_mod(Dataset):
    def __init__(self,video_directory, sequence_size = 10,  transform = None):
        self.video_directory = video_directory
        self.frame_count_indexes = video_directory[:,1]
        temp = [0]
        temp.extend(self.frame_count_indexes)
        self.frame_count_indexes = temp
        
        self.sequence_size = sequence_size
        self.transform = transform
    def __len__(self):
        return int(self.video_directory[-1][1]/self.sequence_size)
    def __getitem__(self, idx):
        for i in range(1,len(self.frame_count_indexes)):
            if(self.frame_count_indexes[i-1]<=idx * self.sequence_size<self.frame_count_indexes[i]):
                video_index = i-1
                #frame_no = idx - self.frame_count_indexes[i-1]
                break
        video_path = self.video_directory[video_index,0]
        frames = []
        start_frame = idx * self.sequence_size -  self.frame_count_indexes[video_index] * self.sequence_size
        vidObj = cv2.VideoCapture(video_path)
        for frame in self.frame_extract(start_frame,vidObj):
            if(self.transform):
                frames.append(self.transform(frame))
            else:
                frames.append(frame)
            if(len(frames) == self.sequence_size):
                break
        
        if self.transform is not None: 
            for i in range(30-len(frames)):
                frames.append(self.transform(frame))
        frames = torch.from_numpy(np.asarray(frames))
        #frames = torch.stack(frames)
        vidObj.release()
        cv2.destroyAllWindows()
        return frames
    def frame_extract(self,frame_no,vidObj):
         
        vidObj.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                image = cv2.resize(image,(128,128))
                yield image

class VideoLoader(Dataset):
    def __init__(self,video_directory, sequence_size = 10,  transform = None):
        self.video_directory = video_directory
        self.frame_count_indexes = video_directory[:,1]
        temp = [0]
        temp.extend(self.frame_count_indexes)
        self.frame_count_indexes = temp
        
        self.sequence_size = sequence_size
        self.transform = transform
    def __len__(self):
        return int(self.video_directory[-1][1]/self.sequence_size)

    def __getitem__(self, idx):
        for i in range(1,len(self.frame_count_indexes)):
            if(self.frame_count_indexes[i-1]<=idx<self.frame_count_indexes[i]):
                video_index = i-1
                frame_no = idx - self.frame_count_indexes[i-1]
                break
        video_path = self.video_directory[video_index,0]
        frames = []
        start_frame = abs(frame_no - (self.sequence_size-1)/2)
        for frame in self.frame_extract(video_path,start_frame):
            if(self.transform):
                frames.append(self.transform(frame))
            else:
                frames.append(frame)
            if(len(frames) == self.sequence_size):
                break
        
        if self.transform is not None: 
            for i in range(30-len(frames)):
                frames.append(self.transform(frame))
        frames = torch.from_numpy(np.asarray(frames))
        #frames = torch.stack(frames)
        return frames

    def frame_extract(self,path,frame_no):
        vidObj = cv2.VideoCapture(path) 
        vidObj.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                image = cv2.resize(image,(128,128))
                yield image


def save_generated_img(model,x,number_of_images,epoch,save_path,step):
   
    samples = model(x).detach().cpu().numpy()[:number_of_images]
    samples = (samples * 0.5 + 0.5) * 255
    generated_images = []
    if not(os.path.isdir(save_path + 'saved_images/ae')):
        os.mkdir(save_path + 'saved_images/ae')
    i=0
    print('........ Saving Images ........')
    for sample in samples:
        sample = np.transpose(sample,(1,2,0)).astype(np.uint8)
        i += 1
        io.imsave(save_path + 'saved_images/ae/epoch_' + str(epoch) + '_' + str(step) + '_' + str(i) + '.png', sample)
    return generated_images


def save_model(model,epoch,save_path):

    if not(os.path.isdir(save_path + 'saved_models/ae')):
        os.mkdir(save_path + 'saved_models/ae')
    model_path = save_path + 'saved_models/ae/epoch_' + str(epoch) + '.pt'
    torch.save(model.state_dict(),model_path)
    print('......... Saving Models .........')
