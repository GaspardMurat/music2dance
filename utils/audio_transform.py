import librosa
import numpy as np
import os
import logging

import matplotlib.pyplot as plt

def load_audio(path, samplingrate=44100):
    sound_path = os.path.join(path, 'audio.mp3')

    #    Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(sound_path, sr=samplingrate, dtype='float32')

    return y, sr


# TODO : to apply noise, or other transformation to audiodata, use this function after load_audio().
def audio_augmontation(audiodata, SNR_dB):
    #audiodata /= np.amax(np.abs(audiodata))
    if SNR_dB < 0:
        return audiodata
    else:
        snr = 10.0 ** (SNR_dB / 10.0)
        p1 = audiodata.var()
        n = p1 / snr
        noise = np.random.randn(audiodata.shape[0]) * np.sqrt(n)
        audiodata +=noise
    return audiodata


def sequentialize(audiodata, start_pos, end_pos, sr=44100, slice_length=1764, wlen=256):
    first = int(start_pos * slice_length + slice_length / 2)
    last = int(end_pos * slice_length + slice_length / 2)

    if last - first > audiodata.shape[0]:
        logging.error('The lenght of the sequence is larger thant the lenght of the file...')
        return 0, False
    else:
        acoustic_features = []
        audiodata = audiodata[first:last]
        n = len(audiodata)
        for x in range(0, n - 1, slice_length):
            slice = audiodata[x:slice_length + x]
            stft = librosa.feature.melspectrogram(y=slice, sr=sr, hop_length=wlen, n_mels=128)
            #stft = librosa.stft(y=slice, hop_length=wlen)
            stft = librosa.amplitude_to_db(np.abs(stft))
            acoustic_features.append(stft)
        acoustic_features = np.concatenate(acoustic_features, axis=1)
        return acoustic_features, True


def reshape_acoustic_features(data, start_pos, end_pos):
    n_frames = end_pos - start_pos
    x = data.shape[0]
    y = int(data.shape[1] / n_frames)
    data = np.reshape(data, (n_frames, x, y))
    data = np.expand_dims(data, axis=3)
    return data


def input_loader(data_wave, start_pos, end_pos, config):
    sr = config['sampling_rate']
    slice_length = config['hop_length']
    wlen = config['window_length']

    acoustic_features, bool = sequentialize(data_wave, start_pos, end_pos, sr=sr, slice_length=slice_length, wlen=wlen)
    if not bool:
        return 0, bool
    acoustic_features = reshape_acoustic_features(acoustic_features, start_pos, end_pos)
    audiodata = np.squeeze(acoustic_features)

    return audiodata, bool


def audio_transform(data_audio, config):
    audio_max = 80.
    audio_min = -100.
    config['slope_wav'] = (config['rng_wav'][1] - config['rng_wav'][0]) / (audio_max - audio_min)
    config['intersec_wav'] = config['rng_wav'][1] - config['slope_wav'] * audio_max
    audiodata = data_audio * config['slope_wav'] + config['intersec_wav']
    return audiodata


def audio_silence(config):
    '''

    :param config:
    :return: white noise array.
    '''
    mean = 0
    std = 1.10e-3
    silence_wav = np.random.normal(mean, std, size=config['silence'] * config['sampling_rate']).astype(np.float32)
    #silence_wav /= np.amax(np.abs(silence_wav))
    end = int(config['silence'] * config['fps'] / 2)
    silence, bool = input_loader(silence_wav, 0, end, config)
    return silence
