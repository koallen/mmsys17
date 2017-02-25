import json
import matplotlib.pyplot as plt

with open("../dictionaries/new_corpus.json", "r") as corpus_json:
    corpus = json.load(corpus_json)

length_list = list()
for sentence in corpus:
    length = len(sentence["content"])
    length_list.append(length)

plt.hist(length_list, bins = 50, color="b")
plt.xlabel("Sentence length")
plt.ylabel("Frequency")
plt.show()

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

x = [pos_count / total_count, neu_count / total_count, neg_count / total_count]
y = [0, 1, 2]
y_label = ["Positive", "Neutral", "Negative"]

plt.bar(y, x, color="b", align="center", width=0.5)
plt.xticks(y, y_label)
plt.ylabel("Relative frequency")
plt.show()
