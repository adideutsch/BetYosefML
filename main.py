import sklearn, pandas, numpy, re

import parsing_utils

BETYOSEF_FILENAME = "BetYosefData/BetYosef-AllText.txt"

data = parsing_utils.get_raw_data(BETYOSEF_FILENAME)

print("Hello World!")

print(len(data))
#
# words_frequency = parsing_utils.build_frequency_dict(data)
#
# threshold = 100
# print("%d above 100 instances" % (len(parsing_utils.get_words_above_frequency(words_frequency, threshold))))
#
# sorted_keys = sorted(words_frequency, key=words_frequency.get)
# sorted_keys.reverse()

# for word in sorted_keys[0:100]:
#     if len(word) < 3:
#         print("Word: <%s>, len: %d, Frequency: %d" % (word, len(word), words_frequency[word]))

brackets_occurrences = re.findall("\([^()]{5,50}\)", data[:100000])

for occ in brackets_occurrences:
    print(occ)

