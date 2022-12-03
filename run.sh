prog=/root/repos/dli_del/train_asr-by-w2v2-ft.py
repo_path=facebook/wav2vec2-large-robust-ft-swbd-300h
output_dir=/root/repos/dli_del/data/asr-tem
train_tsv=/root/repos/dli_del/data/wav_split_gold/train2.tsv
eval_tsv=/root/repos/dli_del/data/wav_split_gold/eval2.tsv
use_target_vocab='--use_target_vocab=False'

python $prog $repo_path $output_dir $train_tsv $eval_tsv $use_target_vocab
