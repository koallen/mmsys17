import json
import jieba

def get_words_in_sentences(corpus):
    """
    Read words from corpus (to be fixed)
    """
    word_list = dict()
    for sentence in corpus:
        for word in sentence["contain_words"]:
            if word["word"] not in word_list:
                word_list[word["word"]] = word["word_sentiment"]
            else:
                pass
    return word_list

def get_sentences(corpus):
    """
    Read sentences from corpus
    """
    sentence_list = list()
    sentence_dict = dict()
    for sentence in corpus:
        sentence_list.append(sentence["content"])
        sentence_dict[sentence["content"]] = sentence["sentiment"]
    return sentence_list, sentence_dict

# load data
with open("new_corpus.json") as corpus:
    corpus_json = json.load(corpus)
    words = get_words_in_sentences(corpus_json)
    sentences, sentence_dict = get_sentences(corpus_json)

# count non-neutral words
non_neutral_count = 0
for word, word_sentiment in words.items():
    if word_sentiment != 'z':
        non_neutral_count += 1

# some stats
print("Non neutral word: " + str(non_neutral_count))
print("There are in total " + str(len(sentences)) + " sentences")
print("There are in total " + str(len(words)) + " words")

# simple classification
correct_count = 0

for sentence in sentences:
    word_list = jieba.cut(sentence)
    total = 0
    for word in word_list:
        if word in words:
            if words[word] == 'p':
                total += 1
            elif words[word] == 'n':
                total -= 1
    if total > 0:
        sentence_sentiment = 'p'
        print("Sentence is positive")
    elif total < 0:
        sentence_sentiment = 'n'
        print("Sentence is negative")
    else:
        sentence_sentiment = 'z'
        print("Sentence is neutral")
    if sentence_dict[sentence] == sentence_sentiment:
        correct_count += 1

print("Correctly classified sentences: " + str(correct_count))
print("Accuracy is: " + str(correct_count/len(sentences)))
