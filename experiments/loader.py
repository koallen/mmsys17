import json

class DictionaryLoader:
    init_dictionary = dict()
    final_dictionary = dict()

    def __init__(self):
        # dictionaries = ["dictionaries/ntusd-positive.txt", "dictionaries/ntusd-negative.txt", "dictionaries/new_corpus.json"]
        # dictionaries = ["new_corpus.json", "ntusd-positive.txt", "ntusd-negative.txt"]
        dictionaries = ["../dictionaries/ntusd-positive.txt", "../dictionaries/ntusd-negative.txt"]
        # dictionaries = ["dictionaries/new_corpus.json"]
        self.load_dicts(dictionaries)
        self.generate_dict()

    def load_dicts(self, dictionaries):
        for dictionary in dictionaries:
            suffix = dictionary.split(".")[-1]
            print("Loading " + dictionary)
            if suffix == "txt":
                self.load_txt(dictionary)
            else:
                self.load_json(dictionary)

    def load_txt(self, filename):
        if "positive" in filename:
            p_or_n = "p"
        elif "negative" in filename:
            p_or_n = "n"
        else:
            pass
        with open(filename, "r") as txt_file:
            words = txt_file.readlines()
            for word in words:
                word = word.strip("\n")
                if word not in self.init_dictionary:
                    # initialize dictionary
                    self.init_dictionary[word] = dict()
                    self.init_dictionary[word]["p"] = 0
                    self.init_dictionary[word]["n"] = 0
                    # increment counter
                    self.init_dictionary[word][p_or_n] += 1
                else:
                    pass

    def load_json(self, filename):
        # current_dict = copy.deepcopy(self.init_dictionary)
        with open(filename, "r") as json_file:
            sentences = json.load(json_file)
            for sentence in sentences:
                words = sentence["contain_words"]
                for word in words:
                    actual_word = word["word"]
                    sentiment = word["word_sentiment"]
                    if sentiment != "z":
                        if actual_word not in self.init_dictionary:
                            # initialize dictionary
                            self.init_dictionary[actual_word] = {"p": 0, "n": 0}
                        else:
                            pass
                        # increment counter
                        if sentiment == "p":
                            self.init_dictionary[actual_word]["p"] += 1
                        elif sentiment == "n":
                            self.init_dictionary[actual_word]["n"] += 1
                        else:
                            pass
                    else:
                        pass

    def generate_dict(self):
        for k, v in self.init_dictionary.items():
            if v["p"] > v["n"]:
                self.final_dictionary[k] = "p"
            elif v["p"] < v["n"]:
                self.final_dictionary[k] = "n"

class CorpusLoader:

    def __init__(self):
        filename = "../dictionaries/new_corpus.json"
        self.load(filename)

    def load(self, filename):
        """
        Read sentences from corpus
        """
        with open(filename, "r") as corpus_json:
            corpus = json.load(corpus_json)
        self.sentence_list = list()
        self.sentence_dict = dict()
        for sentence in corpus:
            self.sentence_list.append(sentence["content"])
            self.sentence_dict[sentence["content"]] = sentence["sentiment"]
