from core import vocoder
import soundfile as sf
import argparse
import os.path
import numpy as np
from meldataset import MAX_WAV_VALUE

DEVICE = "cuda:0"
RUN_PATH = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    sr_voc = vocoder.HiFiGAN(
        os.path.join(RUN_PATH, "models", "hifisr"),
        "config_32k", DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    args = parser.parse_args()

    for n in args.name:
        audio, samplerate = sf.read(n)
        audio = audio * MAX_WAV_VALUE
        assert (samplerate == 22050), 'input must be 22.05kHz'

        sr_mix, new_rate = sr_voc.superres(audio, 22050)

        sf.write(os.path.splitext(os.path.basename(n))[0] + '_sr.wav',
            sr_mix.astype(np.int16), new_rate)