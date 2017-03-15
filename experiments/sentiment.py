import json
import jieba
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import itertools
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def svm_classify(sentences, sentence_dict, dictionary):
    # SVM classification
    training_set, training_label, test_set, test_label = gen_data(sentences, sentence_dict, dictionary)
    class_names = ["neutral", "negative", "positive"]
    clf = svm.SVC(kernel='linear')
    clf.fit(training_set, training_label)
    result = clf.predict(test_set)
    correct_count = np.sum(np.equal(result, test_label))

    # confusion matrix
    cnf_matrix = confusion_matrix(test_label, result)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='')
    plt.show()

    print("SVM accuracy: " + str(correct_count/len(test_label)))
    print("SVM F1 score: " + str(f1_score(test_label, result, average='weighted')))

def convert_label(original_list):
    new_list = list()

    for label in original_list:
        if label == "p":
            new_list.append(2)
        elif label == "n":
            new_list.append(1)
        elif label == "z":
            new_list.append(0)
        else:
            print(label)

    return new_list

def simple_classify(sentences, sentence_dict, dictionary):
    # simple classification
    simple_classifier = SimpleClassifier()
    class_names = ["neutral", "negative", "positive"]
    correct_count = 0
    result_list = list()
    correct_list = list()

    # load inverse dict
    with open("../dictionaries/inverse.txt", "r") as inverse_dict:
        words = inverse_dict.readlines()
        inverse = []
        for word in words:
            word = word.strip("\n")
            inverse.append(word)

    for sentence in sentences:
        result, correct_label, result_label = simple_classifier.classify(sentence, sentence_dict[sentence], dictionary, inverse)
        if result is True:
            correct_count += 1
        result_list.append(result_label)
        correct_list.append(correct_label)

    print(len(result_list))
    print(len(correct_list))
    result_array = np.asarray(convert_label(result_list))
    correct_array = np.asarray(convert_label(correct_list))

    # confusion matrix
    cnf_matrix = confusion_matrix(correct_array, result_array)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='')
    plt.show()

    print("Simple algorithm accuracy: " + str(correct_count/len(sentences)))

if __name__ == "__main__":
    # matplotlib
    matplotlib.rcParams.update({'font.size': 16})
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
