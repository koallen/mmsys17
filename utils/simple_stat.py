import json
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

with open("../dictionaries/new_corpus.json", "r") as corpus_json:
    corpus = json.load(corpus_json)

# Sentence length
length_list = list()
for sentence in corpus:
    length = len(sentence["content"])
    length_list.append(length)

count = 0

for length in length_list:
    if length <= 19:
        count += 1

print("<= 20 ", count/len(length_list))

plt.hist(length_list, bins = 50, color="b")
plt.xlabel("Sentence length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Polarity distribution
pos_count = 0
neu_count = 0
neg_count = 0
total_count = 0

for sentence in corpus:
    total_count += 1
    if sentence["sentiment"] == "p":
        pos_count += 1
    elif sentence["sentiment"] == "n":
        neg_count += 1
    elif sentence["sentiment"] == "z":
        neu_count += 1

for polarity in [pos_count, neu_count, neg_count]:
    print("percentage ", polarity * 100.0 / total_count)

x = [pos_count / total_count, neu_count / total_count, neg_count / total_count]
y = [0, 1, 2]
y_label = ["Positive", "Neutral", "Negative"]

#plt.bar(y, x, color="b", align="center", width=0.5)
#plt.xticks(y, y_label)
#plt.xlabel("Sentence polarity")
#plt.ylabel("Relative frequency")
#plt.tight_layout()
#plt.show()

## pie chart
slices = [1,2,3] * 4 + [20, 25, 30] * 2
shuffle(slices)
cmap = plt.cm.prism
colors = cmap(np.linspace(0., 1., len(slices)))
fig1, ax1 = plt.subplots()
ax1.pie(x, labels=y_label, startangle=90, autopct='%1.1f%%', colors=colors)
ax1.axis('equal')
plt.xlabel("Sentence polarity")
plt.show()

# influence-of-speech
total_count = 0
size = [0 for x in range(5)]
for sentence in corpus:
    for word in sentence["contain_words"]:
        if word["word_semantic"] not in ("m", "p", "t"):
            total_count += 1
        if word["word_semantic"] == "e":
            size[3] += 1
        elif word["word_semantic"] == "i":
            size[0] += 1
        elif word["word_semantic"] == "c":
            size[1] += 1
        elif word["word_semantic"] == "l":
            size[2] += 1
        elif word["word_semantic"] == "a":
            size[4] += 1
size = [i / total_count for i in size]
y_label = ["Intelligence", "Character", "Language", "Emotion", "Art"]

## pie chart
slices = [1,2,3] * 4 + [20, 25, 30] * 2
shuffle(slices)
cmap = plt.cm.prism
colors = cmap(np.linspace(0., 1., len(slices)))
fig1, ax1 = plt.subplots()
ax1.pie(size, labels=y_label, startangle=180, autopct='%1.1f%%', colors=colors)
ax1.axis('equal')
plt.xlabel("Word influence")
plt.show()
