import pandas as pd
from tqdm import tqdm
import helpers.asr as asr
import re

# program controls
asr_mapping_file = '/Volumes/Boston/Scratch/dli_del/wav_split_gold/gold_asr_mapping.csv'
pred_output = './output/results.csv'


def post_processing_transcription(string):
    string = string.strip('.«,').replace('=', '').replace(' ', '').replace('Ø', ' ')
    string = re.sub('\(.+?\)', '', string)
    string = string.replace(' ', '')
    string = string.strip(',')
    return string


def post_processing_recognized(string):
    string = string.replace(' ', '')
    string = string if string else '-'
    return string


# read asr golden labels
print("Reading file...")
df_gl = pd.read_csv(asr_mapping_file)

# read in each of the converted files
lof = df_gl['full_path_outfile'].to_list()

# get predictions for each strip
print("Pred text...")
results = []
for f in tqdm(lof):
    try:
        pred = asr.recognizer(f)
    except:
        pred = "ERROR"
    results.append(pred)
df_gl['recognized'] = results

# clean transcription text
df_gl['transcription_clean'] = df_gl['transcription'].apply(post_processing_transcription)

# clean recongized text
df_gl['recognized_clean'] = df_gl['recognized'].apply(post_processing_recognized)
                                            
# save results to csv file
df_gl.to_csv(pred_output)
