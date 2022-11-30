import glob
import os
import pandas as pd
import torch
import torchaudio
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.utils._testing import ignore_warnings
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def get_sli_df(sli_train_dir):
    print('sli_train_dir: {}'.format(sli_train_dir))
    languages = [f for f in os.listdir(sli_train_dir) if not f.startswith('.')]
    lang_dfs = {}

    for l in languages:
        wav_paths = glob.glob(os.path.join(sli_train_dir, l, "*.wav"))
        lang_dfs[l] = pd.DataFrame.from_dict({'wav_path': wav_paths, 'lang': l})

    sli_df = pd.concat(lang_dfs.values(), ignore_index=True)
    return sli_df


def get_sb_encoder(save_dir="tmp"):
    sb_encoder = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=save_dir,
                                                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    return sb_encoder


def add_sbemb_cols(sli_df, sb_encoder):
    wav_path_list = sli_df["wav_path"].to_list()
    lang_list = sli_df["lang"].to_list()
    sbemb = []

    for f in tqdm(wav_path_list):
        waveform, sample_rate = torchaudio.load(f)

        if torchaudio.info(f).num_channels == 2:
            waveform = waveform.sum(axis=0) / 2

        if sample_rate != 16000:
            sample_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = sample_to_16k(waveform)

        emb = sb_encoder.encode_batch(waveform)
        sbemb_list = emb.reshape((1, 256)).cpu().detach().tolist()
        sbemb.append(*sbemb_list)
        lang_list[1] = 'eng'
    return wav_path_list, lang_list, sbemb


def colsplit_feats_labels(sli_df):
    # Split data frame columns, return features and labels separately
    return sli_df.iloc[:, -256:], sli_df.lang


@ignore_warnings(category=ConvergenceWarning)
def get_logreg_f1(train_df, test_df):

    train_feats, train_labels = colsplit_feats_labels(shuffle(train_df))
    test_feats, test_labels = colsplit_feats_labels(test_df)

    logreg = LogisticRegression(class_weight='balanced', max_iter = 1000, random_state=0)
    logreg.fit(train_feats, train_labels)
    test_pred = logreg.predict(test_feats)

    results_dict = classification_report(test_labels, test_pred, output_dict=True, zero_division=0)
    f1 = round(results_dict['weighted avg']['f1-score'], 3)

    return f1, test_pred
