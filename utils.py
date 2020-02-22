import pickle


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def save(model, path='./model.pkl'):
    with open(path, 'wb') as file:  # Overwrites any existing file.
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)


def load(path='./model.pkl'):
    with open(path, 'rb') as file:  # Overwrites any existing file.
        return pickle.load(file)
