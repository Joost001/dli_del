"""
This program uses the VAD program to split the files which have already been converted to wav format.
"""

from helpers.preprocessing import get_files_in_dir
from helpers.vad import interface
from tqdm.auto import tqdm
import os

# get list of file
lof = get_files_in_dir("/Volumes/Boston/Scratch/dli_del/wav/", source_format='.wav')

# set variables
outdir = '/Volumes/Boston/Scratch/dli_del/wav_split_vad'
mapping_file = '/Volumes/Boston/Scratch/dli_del/wav_split_vad/vad_asr_mapping.csv'

if not os.path.exists(outdir):
     os.makedirs(outdir)

# apply VAD to files 
for f in tqdm(lof, position=0):
    interface(f, outdir, mapping_file)
