import librosa
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json

from lib.paramManager import paramManager

class NSynthDataSet(Dataset):
    def __init__(self, data_dir, sr=16000, sample_length=128, split='train'):
        
        self.data_dir = data_dir
        self.sr = sr
        self.sample_length = sample_length
        
        params = {}
        pm = paramManager(self.data_dir, self.data_dir)
        file_list = [ filename for filename in pm.filenames(self.data_dir) if filename.endswith('.params')]
        for  filename in file_list:
            with open(os.path.join(self.data_dir, filename)) as f:
                fileparams = json.load(f)
                filedict = {}
                filedict['amplitude'] = fileparams['amplitude']['values'][0]
                filedict['instrument'] = fileparams['instID']['values'][0]
                filedict['pitch'] = fileparams['midiPitch']['values'][0]
                filedict['normPitch'] = fileparams['normPitch']['values'][0]
                paramname = filename.replace('.params','')
                filedict['filename'] = paramname+'.wav'
                
                params[paramname] = filedict

        self.nsynth_df = pd.DataFrame.from_dict(params).transpose()
        
        total_dupes = 0
        if split == 'train':
            total_dupes = 1
        else:
            total_dupes = 0 #test split
            self.nsynth_df = self.nsynth_df[self.nsynth_df['amplitude']%(0.2) == 0]
        
        self.nsynth_df['fold'] = 0
        nsynth_df_orig = self.nsynth_df.copy(deep=True)
        for dupe_idx in range(1, total_dupes+1):
            nsynth_df_dupe = nsynth_df_orig.copy(deep=True)
            nsynth_df_dupe['fold'] = dupe_idx
            nsynth_df_dupe.index = nsynth_df_dupe.index + f'-{dupe_idx}'
            self.nsynth_df = pd.concat([self.nsynth_df, nsynth_df_dupe])

        self.nsynth_df = self.nsynth_df.sample(frac=1) #Shuffle data
        print('Shape of dataset = ', self.nsynth_df.shape)
        
    
    def __len__(self):
        return self.nsynth_df.shape[0]
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist()

        data = self.nsynth_df.iloc[idx]
        audio, _ = librosa.load(os.path.join(self.data_dir, data['filename']), sr=self.sr)
        start_location = np.random.randint(1, len(audio) - (self.sample_length + 1))
        input_data = audio[start_location : start_location + self.sample_length]
        if data['fold'] == 1:
            input_data = input_data + np.random.normal(0,0.003,input_data.shape)

        input_data = np.clip(input_data, a_max=1.0, a_min=-1.0)
        input_data = (librosa.mu_compress(input_data, quantize=False) + 1)/2 # Bring values from range [-1 to 1] to [0 to 1]
        
        pitch = np.broadcast_to(np.array([data['normPitch']]), input_data.shape).astype(np.float)
        pitch.setflags(write=1) #Fix for weird Pytorch message "The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
        
        scale = np.broadcast_to(np.array([data['amplitude']]), input_data.shape).astype(np.float)
        scale.setflags(write=1) #Fix for weird Pytorch message "The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
        
        instrument = np.broadcast_to(np.array([data['instrument']]), input_data.shape).astype(np.float)
        instrument.setflags(write=1) #Fix for weird Pytorch message "The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
        
        input_data = np.stack((pitch, scale, instrument, input_data), axis=1)
        input_data = input_data.astype(np.float)
        
        target = audio[start_location + 1 : start_location + 1 + self.sample_length]
        target = librosa.mu_compress(target, quantize=True) + 128
        target = target.astype(np.long)
        
        return input_data, target