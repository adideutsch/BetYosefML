ONLY_PRECEDING_WORDS = False

class Reference():
    def __init__(self, index, label, bag_of_words, bag_size):
        self.index = index
        self.label = label
        self.bag_of_words = bag_of_words
        self.bag_size = bag_size
    def get_bag_of_words(self, size=None):
        if size == None:
            size = self.bag_size
        if ONLY_PRECEDING_WORDS:
            return self.bag_of_words[int(len(self.bag_of_words) / 2) - size: int(len(self.bag_of_words) / 2)]
        return self.bag_of_words[int(len(self.bag_of_words) / 2) - size: int(len(self.bag_of_words) / 2) + size]

