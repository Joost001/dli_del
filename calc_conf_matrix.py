import os
import pandas as pd
from helpers.evaluation import f1_score_regional


def set_pd_print_options():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    
def __get_filename__(x):
    return os.path.splitext(os.path.basename(x))[0]


def format_mapping_files(gold_file_csv, pred_file_csv):
    gold_df = pd.read_csv(gold_file_csv)
    assert 'full_path_infile' in gold_df.columns, "missing column full_path_infile"
    gold_df['filename'] = gold_df['full_path_infile'].apply(__get_filename__)
    assert "start" in gold_df.columns, "missing column start"
    assert "end" in gold_df.columns, "missing column end"
    
    pred_df = pd.read_csv(pred_file_csv)
    assert 'infile' in pred_df.columns, "missing column infile"
    pred_df['filename'] = pred_df['infile'].apply(__get_filename__)
    assert "start_sec" in pred_df.columns, "missing column start_sec"
    assert "end_sec" in pred_df.columns, "missing column end_sec"
    pred_df['start'], pred_df['end'] = pred_df['start_sec'], pred_df['end_sec']
    return gold_df, pred_df

g = "/Volumes/Boston/Scratch/dli_del/wav_split_gold/asr_mapping.csv"
p = "/Volumes/Boston/Scratch/dli_del/wav_split_vad/vad_mapping.csv"

gold, pred = format_mapping_files(g, p)

test = f1_score_regional(gold, pred)
test.to_csv('./output/conf_matrix_parts_by_file.csv')

print('Done')