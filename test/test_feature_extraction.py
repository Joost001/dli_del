import sys
sys.path.insert(0, '/Users/joost/Documents/Repositories/nlp_research/dli_del')
from  helpers.feature_extraction import mfcc_transform, plot_spectrogram, mel_spectrogram
import torchaudio
from torchaudio.utils import download_asset
import unittest

SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

# test mfcc
melkwargs={
        "n_fft": 2048,
        "n_mels": 256,
        "hop_length": 512,
        "mel_scale": "htk"}

args={"sample_rate":6000, "n_mfcc":256, "melkwargs":melkwargs}

mfcc = mfcc_transform(SPEECH_WAVEFORM, args)
plot_spectrogram(mfcc[0])

# test 
args = {
        "sample_rate":6000,
        "n_fft":1024,
        "win_length":None,
        "hop_length":512,
        "center":True,
        "pad_mode":"reflect",
        "power":2.0,
        "norm":"slaney",
        "onesided":True,
        "n_mels":128,
        "mel_scale":"htk"}

melspec = mel_spectrogram(SPEECH_WAVEFORM, args)
plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")

class Testing(unittest.TestCase):
    def test_string(self):
        a = 'some'
        b = 'some'
        self.assertEqual(a, b)

    def test_boolean(self):
        a = True
        b = True
        self.assertEqual(a, b)

if __name__ == '__main__':
    unittest.main()