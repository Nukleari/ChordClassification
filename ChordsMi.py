import os
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import Chords

def preprocess(dfs):
    upper_limit = 1.0e7
    for df in dfs:
        df = df.apply(lambda x : min(x, upper_limit) / upper_limit).astype(np.float16)
    return dfs

def train_and_eval():
    samples = []
    for ch in os.listdir('Audio_Files/major'):
        samples.append([f'Audio_Files/major/{ch}', 'Maj'])

    for ch in os.listdir('Audio_Files/minor'):
        samples.append([f'Audio_Files/minor/{ch}', 'Min'])

    import random
    random.shuffle(samples)

    x = []
    y = []

    for index, sample in enumerate(samples):
        df, ch = Chords.chord_from_file(sample[0], 100, -500)

        x.append(df)
        y.append(sample[1])

    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, random_state=1)

    model = MLPClassifier(max_iter=1000, learning_rate='invscaling')
    model.fit(preprocess(Xtrain), ytrain)

    y_model = model.predict(preprocess(Xtest))
    print(accuracy_score(ytest, y_model))

    return(model)


if __name__ == "__main__":
    model = train_and_eval()
    from joblib import dump, load
    dump(model, 'model.joblib')
