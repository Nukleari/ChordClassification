import numpy as np
import pandas
from cv2 import threshold
from scipy.fft import *
from scipy.io import wavfile

notes_df = pandas.read_csv('notes.csv', index_col=0)


def read_file(file, start_time=0, end_time=-1):
    # Open the file and convert to mono
    sample_rate, data = wavfile.read(file)
    if data.ndim > 1:
        data = data[:, 0]
    else:
        pass

    # Return a slice of the data from start_time to end_time
    dataToRead = data[int(start_time * sample_rate / 1000): int(end_time * sample_rate / 1000) + 1]

    return sample_rate, dataToRead


def fourier_transform(sample_rate, data):
    # Fourier Transform
    N = len(data)
    yf = rfft(data)
    xf = rfftfreq(N, 1 / sample_rate)

    return xf, yf


def note_amplitudes(xf, yf):
    # Find the amplitude of the frequencies
    df = notes_df.applymap(lambda x: np.abs(yf[np.abs(xf - x).argmin()]))
    return df


def generate_chords_df():
    notes = list(notes_df.columns)
    chords = {}
    # Major chords (0-4-7)
    for index, note in enumerate(notes):
        chords[note] = [notes[index],
                        notes[(index+4) % 12], notes[(index+7) % 12]]
    # minor chords (0-3-7)
    for index, note in enumerate(notes):
        chords[f'{note}m'] = [notes[index],
                              notes[(index+3) % 12], notes[(index+7) % 12]]

    return pandas.DataFrame(chords).transpose()


chords_df = generate_chords_df()

threshold = .3e7


def find_chord(df):
    # threshold = df.max()/2
    notes = list(df[df > threshold].index)
    chs = chords_df.index[chords_df.apply(
        lambda x: all(item in notes for item in x), axis=1)]
    if len(chs) == 1:
        return chs[0]
    else:
        return f'?{len(chs)}?'


def chord_from_file(file, start_time=0, end_time=-1):
    sample_rate, data = read_file(file, start_time, end_time)
    xf, yf = fourier_transform(sample_rate, data)
    df = note_amplitudes(xf, yf)
    df = df.apply(np.mean, axis=0)
    return df, find_chord(df)
