from snownlp import tag
import json

with open("../dictionaries/new_corpus.json", "r") as corpus_json:
    corpus = json.load(corpus_json)

training_filename = "ios.txt"
marshal_filename = "ios.marshal"

with open(training_filename, "w") as training_text:
    for sentence in corpus:
        for word in sentence["contain_words"]:
            training_text.write(word["word"] + "/" + word["word_semantic"] + " ")
        training_text.write("\n")

tag.train(training_filename)
tag.save(marshal_filename)
