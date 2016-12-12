import json
import jieba
from sklearn import svm
import numpy as np
from loader import DictionaryLoader
from classifier import SimpleClassifier

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

    # svm
    with open("new_corpus.json", "r") as corpus_json:
        corpus = json.load(corpus_json)
    dict_list = list(dictionary.keys())
    data = list()
    label = list()
    for sentence in corpus:
        word_l = list()
        features = list()
        for word in sentence["contain_words"]:
            word_l.append(word["word"])
        for word in dict_list:
            if word in word_l:
                features.append(1)
            else:
                features.append(0)
        if sentence["sentiment"] == "p":
            label.append(1)
        elif sentence["sentiment"] == "n":
            label.append(2)
        else:
            label.append(0)
        data.append(features)

    # for features in data:
        # for feature in features:
            # print(feature, end=" ")
        # print()

    training_set = np.asarray(data[:3586])
    training_label = np.asarray(label[:3586])
    test_set = np.asarray(data[3586:])
    test_label = np.asarray(label[3586:])
    clf = svm.SVC()
    clf.fit(training_set, training_label)
    result = clf.predict(test_set)
    correct_count = np.sum(np.equal(result, test_label))
    print("Accuracy: " + str(correct_count/len(test_label)))

    # simple classification
    simple_classifier = SimpleClassifier()
    correct_count = 0

    for sentence in sentences:
        result = simple_classifier.classify(sentence, sentence_dict[sentence], dictionary)
        if result is True:
            correct_count += 1
        else:
            pass

    print("Correctly classified sentences: " + str(correct_count))
    print("Accuracy is: " + str(correct_count/len(sentences)))
