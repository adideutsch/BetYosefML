import operator, re, pickle, os.path

import pandas

CACHE_FILENAME = "parsing_cache.pickle"

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

def is_varient(word_a, word_b):
    return word_a == word_b[:-1] or word_a == word_b[1:]

def are_similar(word_a, word_b):
    return is_varient(word_a, word_b) or is_varient(word_b, word_a) or word_a == word_b

def zero_list_maker(n):
    return [0] * n

def load_pickle():
    # If file doesn't exist create initial one
    if not os.path.isfile(CACHE_FILENAME):
        with open(CACHE_FILENAME, 'wb') as fobj:
            pickle.dump({}, fobj, pickle.HIGHEST_PROTOCOL)

    with open(CACHE_FILENAME, 'rb') as fobj:
        data = pickle.load(fobj)
    return data

def dump_pickle(data):
    with open(CACHE_FILENAME, 'wb') as fobj:
        pickle.dump(data, fobj, pickle.HIGHEST_PROTOCOL)

def load_from_cache(call_id):
    cache = load_pickle()
    if call_id in cache:
        return cache[call_id]
    else:
        return False

def dump_to_cache(call_id, result):
    cache = load_pickle()
    cache[call_id] = result
    dump_pickle(cache)

def parse_data_to_matrices(data, minimum_label_frequency_percentage, minimum_label_frequency, bag_size, testset_factor):

    call_id = (data, minimum_label_frequency_percentage, minimum_label_frequency, bag_size, testset_factor)
    cache_data =  load_from_cache(call_id)
    if cache_data != False:
        return cache_data

    # ORGANIZING THE DATASET
    remaining = data

    brackets_occurrences = re.findall("\([^()]*\)", remaining)

    print("%d references" % (len(brackets_occurrences)))

    reference_indexes = {}
    words = []

    ref_raw_data = " ".join(map(lambda x: x.strip("()"), brackets_occurrences))
    reference_frequency = build_frequency_dict(ref_raw_data)
    sorted_keys = sorted(reference_frequency, key=reference_frequency.get)
    sorted_keys.reverse()

    acknowledged_sources = []
    for word in sorted_keys[:int(len(sorted_keys) * minimum_label_frequency_percentage)]:
        if len(word) > 3 and word in reference_frequency and reference_frequency[word] > minimum_label_frequency:
            print("Word: <%s>, len: %d, Frequency: %d" % (word, len(word), reference_frequency[word]))
            acknowledged_sources.append(word)

    invalids = 0
    for index, occ in enumerate(brackets_occurrences):
        # print("calculate occurrence %d of %d (reference_indexes size is %4f)" % (index, len(brackets_occurrences), float(len(list(reference_indexes.keys())))/(index+1)))
        occ_index = remaining.find(occ)
        before = remaining[:occ_index]
        for word in before.split(" "):
            if word != "":
                words.append(word)
        found_sources = []
        for source in acknowledged_sources:
            if source in occ:
                found_sources.append(source)
        if len(found_sources) == 1:
            reference_indexes[len(words)] = (occ, found_sources[0])
        else:
            invalids += 1
        remaining = remaining[occ_index + len(occ):]


    example = list(reference_indexes.keys())[0]
    print("Example #1: %s, at %d, surrounded with: %s" % (reference_indexes[example][1], example, " ".join(words[example-bag_size : example+bag_size])))

    print("%d entries!" % (len(reference_indexes.keys())))

    entries = []
    for ref in reference_indexes.keys():
        index = ref
        label = reference_indexes[index][1]
        bag_of_words = words[index - bag_size : index + bag_size]
        entries.append([index, label, bag_of_words])

    print("Entry example: %s" % (str(entries[0])))

    ref_words = set()
    ref_labels = set()
    for entry in entries:
        for word in entry[2]:
            ref_words.add(word)
        ref_labels.add(entry[1])

    ref_words = list(ref_words)
    ref_labels = list(ref_labels)

    unique_words = len(ref_words)

    print("Out of %d entries we have %d unique words!" % (len(entries), unique_words))
    print("Out of %d original labels we have %d unique labels!" % (len(acknowledged_sources), len(ref_labels)))

    for entry in entries:
        bag_of_words_vector = zero_list_maker(unique_words)
        for index, word_a in enumerate(ref_words):
            for word_b in entry[2]:
                if word_a == word_b:
                    bag_of_words_vector[index] = 1
        entry.append(bag_of_words_vector)
        if sum(bag_of_words_vector) > bag_size * 2:
            print("at %d got %d" % (entry[0], sum(bag_of_words_vector)))

    dataset = pandas.Series(map(lambda x: x[-1], entries))
    print("%s entries!" % (len(dataset)))

    labels_list = list(map(lambda x: ref_labels.index(x[1]), entries))
    labels = pandas.Series(labels_list)
    print("labels:")
    print("%s" % (", ".join(map(str, list(enumerate(ref_labels))))))

    train_size = int(len(dataset) * testset_factor)
    dataset = list(map(lambda x: pandas.Series(x), dataset))

    train_dataset = dataset[:train_size]
    train_labels = labels[:train_size]
    test_dataset = dataset[train_size:]
    test_labels = labels[train_size:]

    result = (train_dataset, train_labels, test_dataset, test_labels, labels_list)
    dump_to_cache(call_id, result)

    return result