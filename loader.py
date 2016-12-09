import json

class DictionaryLoader:
    init_dictionary = dict()
    final_dictionary = dict()

    def __init__(self):
        dictionaries = ["ntusd-positive.txt", "ntusd-negative.txt", "new_corpus.json"]
        # dictionaries = ["ntusd-positive.txt", "ntusd-negative.txt"]
        # dictionaries = ["new_corpus.json"]
        self.load_dicts(dictionaries)
        self.generate_dict()

    def load_dicts(self, dictionaries):
        for dictionary in dictionaries:
            suffix = dictionary.split(".")[-1]
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
                else:
                    pass
                # increment counter
                self.init_dictionary[word][p_or_n] += 1

    def load_json(self, filename):
        with open(filename, "r") as json_file:
            sentences = json.load(json_file)
            for sentence in sentences:
                words = sentence["contain_words"]
                for word in words:
                    actual_word = word["word"]
                    if actual_word not in self.init_dictionary:
                        # initialize dictionary
                        self.init_dictionary[actual_word] = dict()
                        self.init_dictionary[actual_word]["p"] = 0
                        self.init_dictionary[actual_word]["n"] = 0
                    else:
                        pass
                    if word["word_sentiment"] == "p":
                        self.init_dictionary[actual_word]["p"] += 1
                    elif word["word_sentiment"] == "n":
                        self.init_dictionary[actual_word]["n"] += 1

    def generate_dict(self):
        for k, v in self.init_dictionary.items():
            if v["p"] >= v["n"]:
                self.final_dictionary[k] = "p"
            else:
                self.final_dictionary[k] = "n"

