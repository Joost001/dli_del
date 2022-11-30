"""
This program converts all the files located in: /Volumes/Boston/Data/NLP/SharedTask/ASR/Pilot_Data/audio_to_release
from mp4 format into wav format.
There is no splitting
"""

import os
from tqdm import tqdm
from helpers.preprocessing import ffmpeg, split_filename, mkdir_if_dne


def convert_dir(source_dir: str, target_dir: str, source_format=''):
    for path, sub_dirs, files in os.walk(source_dir):
        for name in tqdm(files):
            if name.endswith(source_format):
                source_file = os.path.join(path, name)
                target_file = target_dir + source_file.removeprefix(source_dir).replace('.mp4', '.wav')
                subdir = split_filename(target_file)['dir_name']
                log_dir = subdir + '/logs/'
                log_file = log_dir + name.replace('.mp4', '.log')
                mkdir_if_dne(subdir)
                mkdir_if_dne(log_dir)
                ffmpeg(source_file, target_file, log_file)


source = '/Volumes/Boston/Data/NLP/SharedTask/ASR/Pilot_Data/audio_to_release'
target = '/Volumes/Boston/Scratch/dli_del/wav'
convert_dir(source, target, source_format='mp4')
