import os
import numpy as np
import pandas as pd

# set pd view options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def stratify(df, group, frac):
    group_list = list(df_tsv[group].unique())
    train = []
    test = []
    for strata in group_list:
        df_tmp = df[df[group] == strata].copy()
        test_tmp = df_tmp.sample(frac=frac, replace=False)
        train_tmp = df_tmp.drop(test_tmp.index)
        train.append(train_tmp)
        test.append(test_tmp)
    return pd.concat(train), pd.concat(test)


if __name__ == '__main__':
    data_dir = './data/processed/Train_1_Data'
    df_asr = pd.read_csv(data_dir + '/asr_mapping_processed.csv')
    df_tsv = df_asr[['full_path_outfile', 'transcription', 'lang']].rename(columns={'full_path_outfile': 'path', 'transcription': 'sentence'})

    df_tsv['sentence'].replace(' ', np.nan, inplace=True)
    df_tsv.dropna(subset=['sentence'], inplace=True)
    df_tsv = df_tsv[df_tsv['path'].apply(lambda x: os.path.exists(x))]

    train, test = stratify(df_tsv, 'lang', .1)
    train.to_csv(data_dir + '/train.tsv', sep='\t')
    test.to_csv(data_dir + '/eval.tsv', sep='\t')
