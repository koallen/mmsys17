from snownlp import SnowNLP
import json

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

if __name__ == "__main__":
    # load data and label
    testing_data, testing_label = load_corpus("dictionaries/new_corpus.json")

    # get the tags
    testing_result = list()
    for sentence in testing_data:
        s = SnowNLP(sentence)
        pairs = dict()
        for tag in s.tags:
            pairs[tag[0]] = tag[1]
        testing_result.append(pairs)

    # performance measurement
    total_labels = 0
    for pairs in testing_label:
        total_labels += len(pairs)

    result_labels = 0
    correct_labels = 0
    for i in range(len(testing_data)):
        true_pairs = testing_label[i]
        result_pairs = testing_result[i]
        result_labels += len(result_pairs)
        for word, ios in result_pairs.items():
            if word in true_pairs:
                if true_pairs[word] == ios:
                    correct_labels += 1

    precision = correct_labels / total_labels
    recall = correct_labels / result_labels

    print(precision, recall)

