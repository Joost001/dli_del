from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch
import torchaudio

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# predict string
def recognizer(fpath):
    """Predicts a seq of phonemes (IPA) as a string

    Args:
        fpath (str): full file path of wav file

    Returns:
        string: prediction 
    """
    waveform, sample_rate = torchaudio.load(fpath)
    waveform = waveform.to(device)
    logits = model(waveform).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]
