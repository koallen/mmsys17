import json
import jieba
from sklearn import svm
import numpy as np
from loader import DictionaryLoader, CorpusLoader
from classifier import SimpleClassifier

if __name__ == "__main__":
    # load dictionary and corpus
    dictionary = DictionaryLoader().final_dictionary
    corpus = CorpusLoader()
    sentences = corpus.sentence_list
    sentence_dict = corpus.sentence_dict
    print("Dictionaries and corpus loaded successfully")

    # svm
    # with open("dictionaries/new_corpus.json", "r") as corpus_json:
        # corpus = json.load(corpus_json)
    # dict_list = list(dictionary.keys())
    # data = list()
    # label = list()
    # for sentence in sentences:
        # word_l = list()
        # features = list()
        # for word in jieba.cut(sentence):
            # word_l.append(word)
        # for word in dict_list:
            # if word in word_l:
                # features.append(1)
            # else:
                # features.append(0)
        # if sentence_dict[sentence] == "p":
            # label.append(1)
        # elif sentence_dict[sentence] == "n":
            # label.append(2)
        # else:
            # label.append(0)
        # data.append(features)

    # # for features in data:
        # # for feature in features:
            # # print(feature, end=" ")
        # # print()

    # training_set = np.asarray(data[:3586])
    # training_label = np.asarray(label[:3586])
    # test_set = np.asarray(data[3586:])
    # test_label = np.asarray(label[3586:])
    # clf = svm.SVC()
    # clf.fit(training_set, training_label)
    # result = clf.predict(test_set)
    # correct_count = np.sum(np.equal(result, test_label))
    # print("Accuracy: " + str(correct_count/len(test_label)))

    # simple classification
    simple_classifier = SimpleClassifier()
    correct_count = 0

    #load inverse dict
    with open("dictionaries/inverse.txt") as inverse_dict:
        words = inverse_dict.readlines()
        inverse = []
        for word in words:
            word = word.strip("\n")
            inverse.append(word)

    for sentence in sentences:
        result = simple_classifier.classify(sentence, sentence_dict[sentence], dictionary, inverse)
        if result is True:
            correct_count += 1
        else:
            pass

    print("Correctly classified sentences: " + str(correct_count))
    print("Accuracy is: " + str(correct_count/len(sentences)))
