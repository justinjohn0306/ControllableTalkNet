import os
import pathlib
import base64
import torch
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import io
from nemo.collections.tts.models import TalkNetSpectModel
from nemo.collections.tts.models import TalkNetPitchModel
from nemo.collections.tts.models import TalkNetDursModel
from core.talknet_singer import TalkNetSingerModel
from core import extract, vocoder, reconstruct
from core.download import download_from_drive, download_reconst
import os
import json
import ffmpeg
import traceback
import librosa
import soundfile
import gc
from flask import Flask, jsonify, request

app = Flask(__name__)

DEVICE = "cuda:0"
RUN_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_MELS = True

# Assumes ref audio, no metallic noise reduction, no autotune
class AltTalknetServer:
    def __init__(self):
        self.extract_dur = extract.ExtractDuration(RUN_PATH, DEVICE)
        self.extract_pitch = extract.ExtractPitch()

        self.model_mapping = {}
        self.wav_f0s = {}
        self.tnmodel = None
        self.tnpath = ""
        self.tndurs = None
        self.tnpitch = None
        self.voc = None
        self.last_voc = ""
        self.sr_voc = None
        self.init_model_mapping()
        self.output_mels = False

        #self.from_file0("test_file.txt")

        pass

    def from_file0(self, fil):
        wav_list = []
        lines = []
        with open(fil) as f:
            for line in f.readlines():
                ls = line.split('|')
                print(len(ls))
                wav_list.append(ls[0])
                lines.append(ls)
        for w in wav_list:
            self.preprocess_wav(w)
        for l in lines:
            gen = self.generate_audio(l[1],l[0],l[2])
            with(open(gen[2],"wb")) as f:
                f.write(gen[0])

    # Build model mapping from list
    def init_model_mapping(self):
        for filename in os.listdir("model_lists"):
            if len(filename) < 5 or filename[-5:].lower() != ".json":
                continue
            with open(os.path.join("model_lists", filename)) as f:
                j = json.load(f)
                for source in j:
                    self.map_source(source)
        pass

    def map_source(self, source):
        for char in source["characters"]:
            self.model_mapping[char["name"]] = char["drive_id"]
        pass

    # Pitch analysis
    def preprocess_wav(self, wav_path, pitch_factor=0):
        if not os.path.exists(os.path.join(RUN_PATH, "temp")):
            os.mkdir(os.path.join(RUN_PATH, "temp"))
        ffmpeg.input(os.path.join(RUN_PATH, wav_path)).output(
            os.path.join(RUN_PATH, "temp", os.path.basename(wav_path) + "_conv.wav"),
            ar="22050",
            ac="1",
            acodec="pcm_s16le",
            map_metadata="-1",
            fflags="+bitexact",
        ).overwrite_output().run(quiet=True)
        f0_with_silence, f0_wo_silence = self.extract_pitch.get_pitch(
            os.path.join(RUN_PATH, "temp", os.path.basename(wav_path) + "_conv.wav"),
            legacy=True,
        )
        f0_factor = np.power(np.e, (0.0577623 * float(pitch_factor)))
        f0_with_silence = [x * f0_factor for x in f0_with_silence]
        f0_wo_silence = [x * f0_factor for x in f0_wo_silence]

        self.wav_f0s[wav_path] = {
            "f0_with_silence": f0_with_silence, 
            "f0_wo_silence": f0_wo_silence}
        pass

    # Audio generation
    def generate_audio(self, char, wav_name, transcript,
        disable_reference_audio = False):
        if transcript is None or transcript.strip() == "":
            raise ValueError("Empty transcript")
        pass
        load_error, talknet_path, vocoder_path = download_from_drive(
            self.model_mapping[char], None, RUN_PATH)
        if load_error is not None:
            raise ValueError(load_error)
        with torch.no_grad():
            if self.tnpath != talknet_path:
                singer_path = os.path.join(
                    os.path.dirname(talknet_path), "TalkNetSinger.nemo"
                )
                if os.path.exists(singer_path):
                    print("=== USING SINGER MODEL ===")
                    tnmodel = TalkNetSingerModel.restore_from(singer_path)
                else:
                    tnmodel = TalkNetSpectModel.restore_from(talknet_path)
                durs_path = os.path.join(
                    os.path.dirname(talknet_path), "TalkNetDurs.nemo"
                )
                pitch_path = os.path.join(
                    os.path.dirname(talknet_path), "TalkNetPitch.nemo"
                )
                if os.path.exists(durs_path):
                    tndurs = TalkNetDursModel.restore_from(durs_path)
                    tnmodel.add_module("_durs_model", tndurs)
                    tnpitch = TalkNetPitchModel.restore_from(pitch_path)
                    tnmodel.add_module("_pitch_model", tnpitch)
                else:
                    tndurs = None
                    tnpitch = None
                tnmodel.eval()
                tnpath = talknet_path

        token_list, tokens, arpa = self.extract_dur.get_tokens(transcript)
        if disable_reference_audio:
            if tndurs is None or tnpitch is None:
                print("Error: Model has no pitch predictor;"
                 " cannot generate without reference audio.")
                return [None, None, None, None]
            spect = tnmodel.generate_spectrogram(tokens=tokens)
        else:
            durs = self.extract_dur.get_duration(os.path.basename(wav_name),
                transcript, token_list)

            wav_f0_info = self.wav_f0s[wav_name]["f0_with_silence"]

            spect = tnmodel.force_spectrogram(
                tokens=tokens,
                durs=torch.from_numpy(durs)
                .view(1, -1)
                .type(torch.LongTensor)
                .to(DEVICE),
                f0=torch.FloatTensor(wav_f0_info).view(1, -1).to(DEVICE),
            )

        # Vocoding
        if self.last_voc != vocoder_path:
            self.voc = vocoder.HiFiGAN(vocoder_path, "config_v1", DEVICE)
            self.last_voc = vocoder_path
        audio, audio_torch = self.voc.vocode(spect)

        # Super-resolution
        if self.sr_voc is None:
            self.sr_voc = vocoder.HiFiGAN(
                os.path.join(RUN_PATH, "models", "hifisr"), "config_32k", DEVICE
            )
        sr_mix, new_rate = self.sr_voc.superres(audio, 22050)

        # Create buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, new_rate, sr_mix.astype(np.int16))

        output_name = pathlib.Path(os.path.basename(wav_name)).stem + "_"+char+"tn"

        mel = spect.to('cpu').squeeze().detach().numpy().transpose()
        return [buffer.getvalue(), arpa, output_name, mel]

ats = AltTalknetServer()

@app.route('/characters', methods=['GET'])
def get_characters():
    return jsonify(ats.model_mapping)

@app.route('/upload', methods=['POST'])
def post_audio():
    json_data = request.get_json()
    ats.preprocess_wav(json_data["wav"],
        pitch_factor=json_data.get("transpose",0))
    dra = json_data.get("disable_reference_audio", False)
    data, arpa, name, mel = ats.generate_audio(
        json_data["char"],json_data["wav"],json_data["transcript"],
        disable_reference_audio = dra)
    output_wav = str(pathlib.Path(
        os.path.join(json_data["results_dir"],name)).with_suffix('.wav'))

    if OUTPUT_MELS:
        output_mel = str(pathlib.Path(
            os.path.join(json_data["results_dir"],name)).with_suffix('.npy'))
        np.save(output_mel, mel)

    i = 1
    while os.path.exists(output_wav):
        output_wav = str(pathlib.Path(
            os.path.join(json_data["results_dir"],name)+str(i)).with_suffix('.wav'))
        i += 1
    with open(output_wav,'wb') as f:
        f.write(data)

    gc.collect()
    torch.cuda.empty_cache()
    return jsonify({"output_path":output_wav,"arpabet":arpa})
    
if __name__ == '__main__':
    app.run(debug=True, port=8050)
