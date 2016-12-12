import json
import jieba
from loader import DictionaryLoader
from classifier import SentimentClassifier

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
    # load dictionary
    dictionary = DictionaryLoader().final_dictionary
    print("Dictionary has " + str(len(dictionary)) + " words")
    sentences, sentence_dict = get_sentences("new_corpus.json")

    # simple classification
    classifier = SentimentClassifier()
    correct_count = 0

    for sentence in sentences:
        result = classifier.simple(sentence, sentence_dict[sentence], dictionary)
        if result is True:
            correct_count += 1
        else:
            pass

    print("Correctly classified sentences: " + str(correct_count))
    print("Accuracy is: " + str(correct_count/len(sentences)))
