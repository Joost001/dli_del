"""
This program uses the asr_mapping file to SPLIT AND CONVERT FILES BASED ON HUMAN ANNOTATION.
"""
import os
import pandas as pd
from helpers.preprocessing import ffmpeg_timestamps, mkdir_if_dne
from tqdm import tqdm
import multiprocessing as mp

# set pd view options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# env variables
project_dir = '/Users/joost/Repositories/nlp_research/dli_del'
raw_data_dir = project_dir + '/data/raw/Train_1_Data'
processed_data_dir = project_dir + '/data/processed/Train_1_Data'


# make dir if dne
def mk_dirs():
    mkdir_if_dne(processed_data_dir + '/wav')
    mkdir_if_dne(processed_data_dir + '/logs')


# read in the split information
def read_asr():
    print('Reading asr csv file...')
    df_asr = pd.read_csv(raw_data_dir + '/asr_data.csv')
    return df_asr


# make directory name for the conversion
def mk_work_list(df_asr):
    print('Making filenames for worker...')
    df_asr["full_path_infile"] = raw_data_dir + '/audio_to_release/' + df_asr["lang"].astype(str) + "/" + df_asr[
        "source"].astype(str)
    df_asr['full_path_infile'] = df_asr['full_path_infile'].str.replace(' ', '_')
    df_asr.reset_index(inplace=True)
    df_asr['full_path_outfile'] = df_asr['index'].apply(lambda x: processed_data_dir + '/wav/' + str(x) + '.wav')
    df_asr['log_file_name'] = df_asr['index'].apply(lambda x: processed_data_dir + '/logs/' + str(x) + '.log')

    # save mapping
    df_asr.to_csv(processed_data_dir + '/asr_mapping_processed.csv')

    # make to-do list
    todo_list = df_asr[['full_path_infile', 'full_path_outfile', 'start', 'end', 'log_file_name']]

    # remove items that have been converted already
    out_paths = list(todo_list["full_path_outfile"])
    done = []
    for i in out_paths:
        if os.path.exists(i):
            done.append(i)
    todo_list = todo_list[~todo_list['full_path_outfile'].isin(done)].to_dict('records')

    return todo_list


def process(file):
    # convert files|we do it like this to make use of multiprocessing
    ffmpeg_timestamps(**file)


def multi_process():
    mk_dirs()
    work_list = mk_work_list(read_asr())
    pool = mp.Pool(mp.cpu_count() - 1)
    results = pool.map(process, [w for w in work_list])
    pool.close()


if __name__ == "__main__":
    multi_process()
