import operator

def get_raw_data(filename):
    with open(filename, "r") as fobj:
        data = fobj.read()
    return data


def get_words_above_frequency(words_frequency, threshold):
    return list(filter(lambda x: words_frequency[x] > threshold, words_frequency.keys()))


def build_frequency_dict(data):
    words_frequency = {}

    for word in data.split(" "):
        if word in words_frequency.keys():
            words_frequency[word] += 1
        else:
            words_frequency[word] = 1

    return words_frequency
