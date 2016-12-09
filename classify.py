import json
import jieba
from loader import DictionaryLoader

def get_sentences(filename):
    """
    Read sentences from corpus
    """
    with open(filename, "r") as corpus_json:
        corpus = json.load(corpus_json)
    sentence_list = list()
    sentence_dict = dict()
    for sentence in corpus:
        sentence_list.append(sentence["content"])
        sentence_dict[sentence["content"]] = sentence["sentiment"]
    return sentence_list, sentence_dict

if __name__ == "__main__":
    dictionary = DictionaryLoader().final_dictionary
    sentences, sentence_dict = get_sentences("new_corpus.json")

    # simple classification
    correct_count = 0

    for sentence in sentences:
        word_list = jieba.cut(sentence)
        total = 0
        for word in word_list:
            if word in dictionary:
                if dictionary[word] == 'p':
                    total += 1
                elif dictionary[word] == 'n':
                    total -= 1
        if total > 0:
            sentence_sentiment = 'p'
            # print("Sentence is positive")
        elif total < 0:
            sentence_sentiment = 'n'
            # print("Sentence is negative")
        else:
            sentence_sentiment = 'z'
            # print("Sentence is neutral")
        if sentence_dict[sentence] == sentence_sentiment:
            correct_count += 1

    print("Correctly classified sentences: " + str(correct_count))
    print("Accuracy is: " + str(correct_count/len(sentences)))
