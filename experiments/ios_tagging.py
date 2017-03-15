from snownlp import SnowNLP
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import json

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

def load_corpus(filename):
    testing_data = list()
    testing_label = list()
    with open(filename, "r") as corpus_json:
        corpus = json.load(corpus_json)
        for i in range(123):
            index = i + 5000
            sentence = corpus[index]
            # collect sentence
            testing_data.append(sentence["content"])
            # collect labels
            pairs = dict()
            for word in sentence["contain_words"]:
                pairs[word["word"]] = word["word_semantic"]
            testing_label.append(pairs)

    return testing_data, testing_label

def convert_label(label):
    pass

if __name__ == "__main__":
    # load data and label
    testing_data, testing_label = load_corpus("../dictionaries/new_corpus.json")

    # get the tags
    testing_result = list()
    for sentence in testing_data:
        s = SnowNLP(sentence)
        pairs = dict()
        for tag in s.tags:
            pairs[tag[0]] = tag[1]
        testing_result.append(pairs)

    # performance measurement
    y = list()
    y_pred = list()
    total_labels = 0
    for pairs in testing_label:
        total_labels += len(pairs)

    result_labels = 0
    correct_labels = 0
    for i in range(len(testing_data)):
        true_pairs = testing_label[i] # label of sentence i
        result_pairs = testing_result[i] # pred of sentence i
        result_labels += len(result_pairs)
        for word, ios in result_pairs.items():
            if word in true_pairs:
                if true_pairs[word] == ios:
                    correct_labels += 1
                y.append(true_pairs[word])
                y_pred.append(ios)

    for i in range(100):
        print(y[i])
        print(y_pred[i])

    recall = correct_labels / total_labels
    precision = correct_labels / result_labels
    f1score = (2 * precision * recall) / (precision + recall)

    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 score: " + str(f1score))

    # confusion matrix
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='')
    plt.show()
