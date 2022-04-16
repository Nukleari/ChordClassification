# Musical Instrument Chord Classification (Audio)

### Contents:
- ``Chords.py``: contains functions for processing audio
- ``CircularPlot.py``: Custom visualization for the results
- ``ChordsMi.py``: Train a model on audio files, saves the model using joblib.
- ``test.ipynb``: demonstrates basic usage and tests the results
- ``LiveAudio.py``: demonstrates usage on live audio

### Usage: 
``Chords.py`` can be used tho detect major or minor chords on audio data, from files or live audio. 

``ChordsMi.py`` can be used to train a classifier to distinguish between major and minor chords, can be used in case the first method can't decide. To use the model created, the preprocessing method in ``ChordsMi.py`` must be called on the data first.

For a detailed demonstration of the steps see test.ipynb, for a simple demonstration on live audio see ``LiveAudio.py``
