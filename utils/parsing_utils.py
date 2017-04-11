import re
from joblib import Parallel, delayed
from random import random
import multiprocessing
from functools import reduce

import pandas

import Reference
from utils.task_utils import timed_task, cached_task

BAG_OF_WORDS = 20
MAX_DATA_SIZE = 10 ** 6

@timed_task
@cached_task
def get_raw_data(filename):
    with open(filename, "r") as fobj:
        data = fobj.read()
    print("Reading... %d characters in data" % len(data))
    return data

@timed_task
@cached_task
def get_words_above_frequency(words_frequency, threshold):
    return list(filter(lambda x: words_frequency[x] > threshold, words_frequency.keys()))

@timed_task
@cached_task
def build_frequency_dict(data):
    words_frequency = {}
    for word in data.split(" "):
        if word in words_frequency.keys():
            words_frequency[word] += 1
        else:
            words_frequency[word] = 1
    return words_frequency

@timed_task
@cached_task
def is_varient(word_a, word_b):
    return word_a == word_b[:-1] or word_a == word_b[1:]

@timed_task
@cached_task
def are_similar(word_a, word_b):
    return is_varient(word_a, word_b) or is_varient(word_b, word_a) or word_a == word_b

def zero_list_maker(n):
    return [0] * n

@timed_task
@cached_task
def get_all_references_as_strings(data):
    return re.findall("\([^()]*\)", data)

@timed_task
@cached_task
def get_sorted_references(brackets_occurrences):
    ref_raw_data = " ".join(map(lambda x: x.strip("()"), brackets_occurrences))
    reference_frequency = build_frequency_dict(ref_raw_data)
    sorted_keys = sorted(reference_frequency, key=reference_frequency.get)
    sorted_keys.reverse()
    return sorted_keys, reference_frequency

@timed_task
@cached_task
def choose_acknowledged_references(sorted_keys,
                                   reference_frequency,
                                   minimum_label_frequency_percentage,
                                   minimum_label_frequency):
    acknowledged_sources = []
    for word in sorted_keys[:int(len(sorted_keys) * minimum_label_frequency_percentage)]:
        if len(word) > 3 and word in reference_frequency and reference_frequency[word] > minimum_label_frequency:
            print("Word: <%s>, len: %d, Frequency: %d" % (word, len(word), reference_frequency[word]))
            acknowledged_sources.append(word)
    return acknowledged_sources

def analyze_reference(occ_index, reference_indexes, acknowledged_sources):
    found_sources = []
    occ_words = reference_indexes[occ_index].strip("()").split(" ")
    for source in occ_words:
        if source in acknowledged_sources:
            found_sources.append(source)
    if len(found_sources) == 1:
        return (occ_index, reference_indexes[occ_index], found_sources[0])
    else:
        None

# @timed_task
# @cached_task
def get_all_relevant_references(data, brackets_occurrences, acknowledged_sources, bag_size):
    reference_indexes = {}
    invalids = 0
    remaining = data
    words = []
    for index, occ in enumerate(brackets_occurrences):
        # occ_words = occ.strip("()").split(" ")
        occ_index = remaining.find(occ)  # O(n)
        before = remaining[:occ_index]  # O(1)
        for word in before.split(" ")[1:-1]:
            words.append(word)
        reference_indexes[len(words)] = occ
        remaining = remaining[occ_index + len(occ):]


    num_cores = multiprocessing.cpu_count()


    final_references = {}
    for reference in list(Parallel(n_jobs=num_cores)(delayed(analyze_reference)(occ_index, reference_indexes, acknowledged_sources) for occ_index in reference_indexes.keys())):
        if reference is not None:
            final_references[reference[0]] = reference[2]

    random_example_index = int(random() * len(list(final_references.keys())))
    example = list(final_references.keys())[random_example_index]
    if random() > 0.8:
        print("Example #%d: at %d: %s (~%s) %s" % (random_example_index,
                                                   example,
                                                   " ".join(words[example - bag_size: example]),
                                                   final_references[example],
                                                   " ".join(words[example: example + bag_size])))
    # print("%d references!" % (len(final_references.keys())))
    return final_references, words

# @timed_task
# @cached_task
def create_references(reference_indexes, words, bag_size):
    references = []
    for ref in reference_indexes.keys():
        index = ref
        label = reference_indexes[index]
        bag_of_words = words[index - BAG_OF_WORDS: index + BAG_OF_WORDS]
        references.append(Reference.Reference(index, label, bag_of_words, bag_size))
    return references

@timed_task
# @cached_task
def get_references_metadata(references):
    ref_words = set()
    ref_labels = set()
    for reference in references:
        for word in reference.bag_of_words:
            ref_words.add(word)
        ref_labels.add(reference.label)

    ref_words = list(ref_words)
    ref_labels = list(ref_labels)
    unique_words = len(ref_words)
    return (ref_words, ref_labels, unique_words)

@timed_task
# @cached_task
def create_ml_dataset(references, ref_words, bag_size):
    ref_word_to_index = {}
    for index, word in enumerate(ref_words):
        ref_word_to_index[word] = index
    for reference in references:
        bag_of_words_vector = zero_list_maker(len(ref_words))
        for word in reference.get_bag_of_words():
            if word in ref_words:
                bag_of_words_vector[ref_word_to_index[word]] = 1
        reference.bag_of_words_vector = bag_of_words_vector
        if sum(bag_of_words_vector) > bag_size * 2:
                print("At %d got %d" % (reference.index, sum(bag_of_words_vector)))
    dataset = pandas.Series(map(lambda reference: reference.bag_of_words_vector, references))
    print("Found %s references!" % (len(dataset)))
    return dataset

@timed_task
# @cached_task
def create_ml_labels(ref_labels, references):
    labels_list = list(map(lambda reference: ref_labels.index(reference.label), references))
    labels = pandas.Series(labels_list)
    print("Labels: %s" % (", ".join(map(str, list(enumerate(ref_labels))))))
    return labels, labels_list

@timed_task
def split_dataset(dataset, labels, testset_factor):
    train_size = int(len(dataset) * testset_factor)


    num_cores = multiprocessing.cpu_count()

    dataset = list(Parallel(n_jobs=num_cores)(delayed(pandas.Series)(x) for x in dataset))

    train_dataset = dataset[:train_size]
    train_labels = labels[:train_size]
    test_dataset = dataset[train_size:]
    test_labels = labels[train_size:]
    return train_dataset, train_labels, test_dataset, test_labels

def build_ml_dataset_subroutine(data, brackets_occurrences, acknowledged_sources, bag_size):
    # Omit all irrelevant references and use only those with the popular labels
    reference_indexes, words = get_all_relevant_references(data, brackets_occurrences, acknowledged_sources, bag_size)
    # Create the reference objects
    references = create_references(reference_indexes, words, bag_size)
    return references

def build_ml_dataset(data, brackets_occurrences, acknowledged_sources, bag_size):

    # X - Check for size, if to big - split
    if len(data) > MAX_DATA_SIZE:
        parts = int(len(data) / MAX_DATA_SIZE) + 1
        part_len = int(len(data) / parts)
        split_data = map(lambda i: data[i * part_len : (i + 1) * part_len], range(parts))
        num_cores = multiprocessing.cpu_count()

        references = reduce(lambda x,y: x + y, Parallel(n_jobs=num_cores)(delayed(build_ml_dataset_subroutine)(part, brackets_occurrences, acknowledged_sources, bag_size) for part in split_data))

    else:
        references = build_ml_dataset_subroutine(data, brackets_occurrences, acknowledged_sources, bag_size)

    # Calculate some metadata regarding the references
    ref_words, ref_labels, unique_words = get_references_metadata(references)
    print("Out of %d references we have %d unique words!" % (len(references), unique_words))
    print("Out of %d original labels we have %d unique labels!" % (len(acknowledged_sources), len(ref_labels)))
    # Create an sklearn-compatible dataset for classification algorithms
    dataset = create_ml_dataset(references, ref_words, bag_size)
    # Create an sklearn-compatible labels vector for classification algorithms
    labels, labels_list = create_ml_labels(acknowledged_sources, references)

    return dataset, labels, labels_list

@timed_task
def parse_data_to_matrices(data, minimum_label_frequency_percentage, minimum_label_frequency, bag_size, testset_factor):

    # Finding all the references
    brackets_occurrences = get_all_references_as_strings(data)
    print("%d references" % (len(brackets_occurrences)))

    # Sorting all the references
    sorted_keys, reference_frequency = get_sorted_references(brackets_occurrences)

    # Choosing acknowledged references (only extremely popular ones)
    acknowledged_sources = choose_acknowledged_references(sorted_keys,
                                                          reference_frequency,
                                                          minimum_label_frequency_percentage,
                                                          minimum_label_frequency)

    dataset, labels, labels_list = build_ml_dataset(data, brackets_occurrences, acknowledged_sources, bag_size)

    # Split the dataset and labels to train set and test set
    train_dataset, train_labels, test_dataset, test_labels = split_dataset(dataset, labels, testset_factor)

    result = (train_dataset, train_labels, test_dataset, test_labels, labels_list)
    return result
