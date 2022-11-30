"""
This program uses the asr_mapping file to SPLIT AND CONVERT FILES BASED ON HUMAN ANNOTATION.
"""
import os
import pandas as pd
from helpers.preprocessing import ffmpeg_timestamps, mkdir_if_dne
from tqdm import tqdm

# set pd view options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# env variables
data_dir = '/Volumes/Boston/Data/NLP/SharedTask/ASR/Pilot_Data'
new_dir = '/Volumes/Boston/Scratch/dli_del/wav_split_gold'
logs_dir = '/Volumes/Boston/Scratch/dli_del/wav_split_gold/logs'

# make dir if dne
mkdir_if_dne(new_dir)
mkdir_if_dne(logs_dir)

# read in the split information
df_asr = pd.read_csv(data_dir + '/asr_data.csv')

# make directory name for the conversion
df_asr["full_path_infile"] = data_dir + '/audio_to_release/' + df_asr["lang"].astype(str) + "/" + df_asr["source"].astype(str)
df_asr['full_path_infile'] = df_asr['full_path_infile'].str.replace(' ', '_')
df_asr.reset_index(inplace=True)
df_asr['full_path_outfile'] = df_asr['index'].apply(lambda x: new_dir + '/' + str(x) + '.wav')
df_asr['log_file_name'] = df_asr['index'].apply(lambda x: logs_dir + '/' + str(x) + '.log')

# save mapping
df_asr.to_csv(new_dir + '/gold_asr_mapping.csv')

# make todo list
todo_list = df_asr[['full_path_infile', 'full_path_outfile', 'start', 'end', 'log_file_name']]

# remove items that have been converted already
outpaths = list(todo_list["full_path_outfile"])
done = []
for i in outpaths:
    if os.path.exists(i):
       done.append(i) 
todo_list = todo_list[~todo_list['full_path_outfile'].isin(done)].to_dict('records')

# convert files
for file in tqdm(todo_list):
    ffmpeg_timestamps(**file)
