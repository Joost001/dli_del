import torch
import torchaudio
import os
import pandas as pd
from os.path import exists


def vad(infile, num_threads=1, sampling_rate=16000, use_unnx=False, return_seconds=False, min_speech_duration_ms=100):
    torch.set_num_threads(num_threads)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  verbose=False,
                                  onnx=use_unnx,
                                  trust_repo=True)
    get_speech_timestamps, read_audio = utils[0], utils[2]
    # read audio file
    wav = read_audio(infile, sampling_rate=sampling_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate,
                                              return_seconds=return_seconds,
                                              min_speech_duration_ms=min_speech_duration_ms)
    return speech_timestamps, wav


def save_audio(full_path_outfile: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    # noinspection PyUnresolvedReferences
    torchaudio.save(full_path_outfile, tensor.unsqueeze(0), sampling_rate)


def get_chunks(speech_timestamps, tensor, infile, out_dir, basename, sampling_rate = 16000):
    chunks = []
    suffix = 0
    for i in speech_timestamps:
        start = i['start']
        end = i['end']
        start_sec = round(start / sampling_rate, 4)
        end_sec = round(end / sampling_rate, 4)
        chunk = tensor[start:end]
        outfile = out_dir + '/' + basename + '_{0}.wav'.format(suffix)
        chunks.append([infile, outfile, start, end, chunk, start_sec, end_sec])
        suffix += 1
    return chunks


def save_chunks(chunks, sampling_rate=16000):
    for c in chunks:
        save_audio(c[1], c[4], sampling_rate)

        
def interface(in_file, out_dir, mapping_file='./vad_mapping.csv', sampling_rate = 16000):
    # get the basename from the infile
    tmp = os.path.basename(in_file)
    base_name = os.path.splitext(tmp)[0]
    # use vad on file
    timestamps, wav = vad(in_file)
    # get chunks
    chunks = get_chunks(timestamps, wav, in_file, out_dir, base_name, sampling_rate)
    # record in log file
    mapping_data = pd.DataFrame(chunks, columns=['infile', 'outfile', 'start', 'end', 'chunk', "start_sec", "end_sec"])
    header_flag = not exists(mapping_file)
    mapping_data[['infile', 'outfile', 'start', 'end', "start_sec", "end_sec"]].to_csv(mapping_file, mode='a', index=False, header=header_flag)
    # save the chunks to the directory
    save_chunks(chunks)
