
import pandas as pd
import os
import matplotlib.pyplot as plt

# work around for depricated Iterable collection
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from abydos import distance


def phonetic_metric(pred, gold):
    """Uses the phonetic edit distance from abydos.distance.
    Doc URL: https://abydos.readthedocs.io/en/latest/_modules/abydos/distance/_phonetic_edit_distance.html
    This is a variation on Levenshtein edit distance, intended for strings in IPA, that compares individual 
    phones based on their featural similarity.

    Args:
        pred (list): a python list of the predicted text 1xn
        gold (list): a python list of the true_transcription 1xn
        
    Return:
        dict: a dict with keys [success, failure] where success is a list containing the edit distances
                between the pred transcription and the true transcription.
    """
    phonetic = distance.PhoneticEditDistance()
    success = []
    failed = []
    assert(len(pred)==len(gold)), "List must be the same length."
    for p, t in pred, gold:
        try:
            success.append(phonetic.dist(p, t))
        except Exception as e:
            failed.append((p, t, e))
    plt.hist(success, bins=50)
    plt.ylabel('Freq')
    plt.xlabel('Edit Distance')
    plt.title('PhoneticEditDistance()')
    plt.savefig('phonetic_ev.png', bbox_inches='tight')
    return {"success": success, "failed": failed}


def __get_conf_matrix__(tuples, filename):
    # init values for loop
    confusion_matrix = []
    gold_vad = 'off'
    pred_vad = 'off'
    last_time = float(0)
    
    # set the value of conf matrix
    for i in tuples:
        if gold_vad == 'on' and pred_vad == 'on': conf = 'TP'
        elif gold_vad == 'off' and pred_vad == 'off': conf = 'TN'
        elif gold_vad == 'on' and pred_vad == 'off': conf = 'FN'
        elif gold_vad == 'off' and pred_vad == 'on': conf = 'FP'
        else: raise Exception("ERROR in states of VAD")
        
        # calc values and store
        time_calc = float(i[0]) - float(last_time)
        confusion_matrix.append((conf, time_calc))
        
        # update VAD states 
        if i[1] == 'start' and i[2] == 'P': pred_vad = 'on'
        elif i[1] == 'start' and i[2] == 'G': gold_vad = 'on'
        elif i[1] == 'end' and i[2] == 'P': pred_vad = 'off'
        elif i[1] == 'end' and i[2] == 'G': gold_vad = 'off'
        else: raise Exception("ERROR in states of VAD")
        
        # update last time
        last_time = i[0]

    tmp1 = pd.DataFrame().from_records(confusion_matrix)
    tmp2 = tmp1.groupby(0, as_index=False)[1].sum()
    tmp2 = tmp2.rename(columns={0:'conf_type', 1:'value'})
    tmp2['filename'] = filename
    return tmp2


def __get_tuples__(gold_i, pred_i):
    list1 =  pd.DataFrame({'time': gold_i['start'], 'cat': 'start', 'type': 'G'})
    list2 =  pd.DataFrame({'time': gold_i['end'], 'cat': 'end', 'type': 'G'})
    list3 =  pd.DataFrame({'time': pred_i['start'], 'cat': 'start', 'type': 'P'})
    list4 =  pd.DataFrame({'time': pred_i['end'], 'cat': 'end', 'type': 'P'})
    w = pd.concat([list1, list2, list3, list4], ignore_index=True)
    w.sort_values('time', ignore_index=True, inplace=True)
    tuples = list(w.itertuples(index=False, name=None))
    return tuples


def f1_score_regional(gold_df, pred_df):
    """This function computes the F1, recall, and precision based off regions of detected voices.
    True Positives: The Gold VAD is on and the Pred VAD is on.
    True Negatives: The Gold VAD is off and the Pred VAD is off.
    False Positives: The Gold VAD is off and the Pred VAD is on.
    False Negatives: The Gold VAD is on and the Pred VAD is off.

    Args:
        gold_df (pd.DataFrame): pd.DataFrame of the timestamps of the gold label VAD
        pred_df (pd.DataFrame): pd.DataFrame of the timestamps of the pred label VAD
    """
    # make a unique list of documents for GOLD and PRED
    gold_file_names = set(gold_df['filename'].tolist())
    pred_file_names = set(pred_df['filename'].tolist())
    work_list = list(pred_file_names.intersection(gold_file_names))
    results = pd.DataFrame()
    for file_i in work_list:
        # make an array of tuples where each tuple is (VAD ID: {G,P}, timestamp id: {S, E}, timestamp: time is seconds)
        gold_i = gold_df[gold_df['filename'] == file_i]
        pred_i = pred_df[pred_df['filename'] == file_i]
        tuples = __get_tuples__(gold_i, pred_i)
        matrix = __get_conf_matrix__(tuples, file_i)
        results = pd.concat([results, matrix], ignore_index=True) 
    return results


# todo create a char error rate
def CER():
    pass


# todo convert phonetic to error rate
def PER():
    pass