import cv2
import numpy as np
import sounddevice as sd
from joblib import dump, load

import Chords
from ChordsMi import preprocess
from CircularPlot import circular_plot_numpy

sample_rate = 44100  # Sample rate
seconds = 1  # Duration of recording
data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
              channels=2, dtype='int16')[:, 0]
sd.wait()  # Wait until recording is finished

model = load('model.joblib')


def callback(indata, outdata, frames, time, status):
    global data
    if status:
        print(status)
    new_data = indata[:, 0]
    data = np.hstack((data[len(new_data):], new_data))


with sd.Stream(samplerate=sample_rate, channels=2, dtype='int16', callback=callback):
    while True:
        xf, yf = Chords.fourier_transform(sample_rate, data)
        df = Chords.note_amplitudes(xf, yf)
        df = df.apply(np.max, axis=0)
        ch = Chords.find_chord(df)
        chm = ''
        if ch[0] == '?':
                ch = ''

        if df[df > Chords.threshold].count() > 2:
            chm = model.predict(preprocess([df]))[0]
            if ch == '':
                ch = chm

        frame = cv2.cvtColor(circular_plot_numpy(df, ch, chm), cv2.COLOR_RGB2BGR)

        cv2.imshow('live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
