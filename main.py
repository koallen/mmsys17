import jieba
from sklearn import svm
import numpy as np
from loader import DictionaryLoader, CorpusLoader
from classifier import SimpleClassifier

def gen_data(sentence, sentence_dict, dictionary):
    """
    Prepares data for SVM classification
    """
    split = 0.7 # split between training and test set

    dict_list = list(dictionary.keys())
    data = list()
    label = list()
    for sentence in sentences:
        word_l = list()
        features = list()
        for word in jieba.cut(sentence):
            word_l.append(word)
        for word in dict_list:
            if word in word_l:
                features.append(1)
            else:
                features.append(0)
        if sentence_dict[sentence] == "p":
            label.append(1)
        elif sentence_dict[sentence] == "n":
            label.append(2)
        else:
            label.append(0)
        data.append(features)

    split_index = int(len(data) * split)
    training_set = np.asarray(data[:split_index])
    training_label = np.asarray(label[:split_index])
    test_set = np.asarray(data[split_index:])
    test_label = np.asarray(label[split_index:])

    return training_set, training_label, test_set, test_label

def svm_classify(sentences, sentence_dict, dictionary):
    # SVM classification
    training_set, training_label, test_set, test_label = gen_data(sentences, sentence_dict, dictionary)
    clf = svm.SVC()
    clf.fit(training_set, training_label)
    result = clf.predict(test_set)
    correct_count = np.sum(np.equal(result, test_label))
    print("SVM accuracy: " + str(correct_count/len(test_label)))

def simple_classify(sentences, sentence_dict, dictionary):
    # simple classification
    simple_classifier = SimpleClassifier()
    correct_count = 0

    #load inverse dict
    with open("dictionaries/inverse.txt", "r") as inverse_dict:
        words = inverse_dict.readlines()
        inverse = []
        for word in words:
            word = word.strip("\n")
            inverse.append(word)

    for sentence in sentences:
        result = simple_classifier.classify(sentence, sentence_dict[sentence], dictionary, inverse)
        if result is True:
            correct_count += 1

    print("Simple algorithm accuracy: " + str(correct_count/len(sentences)))

if __name__ == "__main__":
    # load dictionary and corpus
    dictionary = DictionaryLoader().final_dictionary
    corpus = CorpusLoader()
    sentences = corpus.sentence_list
    sentence_dict = corpus.sentence_dict
    print("Dictionaries and corpus loaded successfully")

    # SVM
    svm_classify(sentences, sentence_dict, dictionary)
    # simple
    simple_classify(sentences, sentence_dict, dictionary)
