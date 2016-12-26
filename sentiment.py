import json
import jieba
from sklearn import svm
from sklearn.metrics import f1_score
import numpy as np
from loader import DictionaryLoader, CorpusLoader
from classifier import SimpleClassifier

def gen_data(sentence, sentence_dict, dictionary):
    """
    Prepares data for SVM classification
    """
    split = 0.7 # split between training and test set

    pos_data = list()
    neg_data = list()
    neu_data = list()
    for sentence in sentences:
        word_l = jieba.lcut(sentence)
        features = list()
        for word in word_list:
            if word in word_l:
                features.append(1)
            else:
                features.append(0)
        if sentence_dict[sentence] == "p":
            pos_data.append(features)
        elif sentence_dict[sentence] == "n":
            neg_data.append(features)
        else:
            neu_data.append(features)
    pos_label = np.full((len(pos_data),), 2)
    neg_label = np.full((len(neg_data),), 1)
    neu_label = np.full((len(neu_data),), 0)

    pos_split_index = int(len(pos_data) * split)
    neg_split_index = int(len(neg_data) * split)
    neu_split_index = int(len(neu_data) * split)

    training_pos_data = np.asarray(pos_data[:pos_split_index])
    training_neg_data = np.asarray(neg_data[:neg_split_index])
    training_neu_data = np.asarray(neu_data[:neu_split_index])
    training_set = np.concatenate((training_pos_data, training_neg_data, training_neu_data))

    training_pos_label = pos_label[:pos_split_index]
    training_neg_label = neg_label[:neg_split_index]
    training_neu_label = neu_label[:neu_split_index]
    training_label = np.concatenate((training_pos_label, training_neg_label, training_neu_label))

    test_pos_data = np.asarray(pos_data[pos_split_index:])
    test_neg_data = np.asarray(neg_data[neg_split_index:])
    test_neu_data = np.asarray(neu_data[neu_split_index:])
    test_set = np.concatenate((test_pos_data, test_neg_data, test_neu_data))

    test_pos_label = pos_label[pos_split_index:]
    test_neg_label = neg_label[neg_split_index:]
    test_neu_label = neu_label[neu_split_index:]
    test_label = np.concatenate((test_pos_label, test_neg_label, test_neu_label))

    print(training_pos_data.shape)
    print(training_neg_data.shape)
    print(training_neu_data.shape)
    print(test_pos_data.shape)
    print(test_neg_data.shape)
    print(test_neu_data.shape)

    return training_set, training_label, test_set, test_label

def svm_classify(sentences, sentence_dict, dictionary):
    # SVM classification
    training_set, training_label, test_set, test_label = gen_data(sentences, sentence_dict, dictionary)
    clf = svm.SVC(kernel='linear')
    clf.fit(training_set, training_label)
    result = clf.predict(test_set)
    correct_count = np.sum(np.equal(result, test_label))
    print("SVM accuracy: " + str(correct_count/len(test_label)))
    print("SVM F1 score: " + str(f1_score(test_label, result, average='weighted')))

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

    with open("dictionaries/new_corpus.json", "r") as corpus_json:
        corpus = json.load(corpus_json)
        word_list = list()
        for sentence in corpus:
            for word in sentence["contain_words"]:
                actual_word = word["word"]
                if actual_word not in word_list:
                    word_list.append(actual_word)

    print(len(word_list))

    # SVM
    svm_classify(sentences, sentence_dict, dictionary)
    # simple
    simple_classify(sentences, sentence_dict, dictionary)
