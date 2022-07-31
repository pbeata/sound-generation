#!/usr/bin/env python
# coding: utf-8

# ## Audio Preprocessing
# 
# 1. load a file
# 2. pad the signal (if necessary)
# 3. extracting log spectrogram from signal
# 4. normalize the spectrogram
# 5. save the normalized spectrogram

# In[27]:


import os
import librosa
import pickle
import numpy as np
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[28]:


class Loader:
    """
    Loader is responsible for loading an audio file.
    """
    
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        
    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal


# In[29]:


class Padder:
    """
    Padder is responsible to apply padding to an array.
    """
    
    def __init__(self, mode="constant"):
        self.mode = mode
    
    def left_pad(self, array, num_missing_items):
        # [1, 2, 3] --> want to pad it with 2 items (pre or post append)
        # e.g. pre-append (left) with zeros --> [0, 0, 1, 2, 3]
        padded_array = np.pad(array, 
                             (num_missing_items, 0),
                             mode=self.mode)
        return padded_array
        
    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, 
                             (0, num_missing_items),
                             mode=self.mode)
        return padded_array


# In[30]:


class LogSpectrogramExtractor:
    """
    LogSpectrogramExtractor extracts log spectrograms (in dB) from a time-series signal.
    """
    
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
        
    def extract(self, signal):
        stft = librosa.stft(signal,
                           n_fft=self.frame_size,
                           hop_length=self.hop_length)[:-1] 
        # (1 + frame_size / 2, num_frames): e.g. 1024 -> 513 -> drop to 512
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


# In[31]:


class MinMaxNormalizer:
    """
    MinMaxNormalizer applies min-max normalization to an array.
    """
    
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
    
    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array
    
    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


# In[39]:


class Saver:
    """
    Saver is responsible for saving features and the min/max values.
    """
    
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
    
    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path
    
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, 
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)
   
    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


# In[58]:


class PreprocessingPipeline:
    """
    PreprocessingPipeline processes audio files in a directory, applying the following steps to each file:
    1. load a file
    2. pad the signal (if necessary)
    3. extracting log spectrogram from signal
    4. normalize the spectrogram
    5. save the normalized spectrogram
    
    **Storing the min-max values for all of the log spectrograms.
    """
    
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
    
    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for audio_file in files:
                audio_path = os.path.join(root, audio_file)
                self._process_file(audio_path)
                print(f"Processed file {audio_path}")
        self.saver.save_min_max_values(self.min_max_values)
    
    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_passing_needed(signal):
            signal = self._apply_padding(signal)
            
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())
    
    def _is_passing_needed(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


# In[59]:


if __name__ == "__main__":

    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74 # in seconds
    SAMPLE_RATE = 22050
    MONO = True
    
    # UNIX-BASED SYSTEMS
#     SPECTROGRAMS_SAVE_DIR = "/home/pbeata/datasets/fsdd/spectrograms/"
#     MIN_MAX_VALUES_SAVE_DIR = "/home/pbeata/datasets/fsdd/"
#     FILES_DIR = "/mnt/c/Users/pbeata/Desktop/Data_Science/Audio/free-spoken-digit-dataset/"
    
    # TEMPORARY FIX FOR WINDOWS
    SPECTROGRAMS_SAVE_DIR = "C:\\Users\\pbeata\\Desktop\\Data_Science\\Audio\\sound-generation\\datasets\\fsdd\\spectrograms\\"
    MIN_MAX_VALUES_SAVE_DIR = "C:\\Users\\pbeata\\Desktop\\Data_Science\\Audio\\sound-generation\\datasets\\fsdd\\"
    FILES_DIR = "C:\\Users\\pbeata\Desktop\\Data_Science\\Audio\\free-spoken-digit-dataset\\recordings\\"    
    
    # instantiate all objects 
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normalizer = min_max_normalizer 
    preprocessing_pipeline.saver = saver
    
    preprocessing_pipeline.process(FILES_DIR)

