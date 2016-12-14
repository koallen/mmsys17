import json

# load old corpus (final_corpus.json)
with open("final_corpus.json") as old_corpus:
    old_corpus_json = json.load(old_corpus)

# reformat the corpus
new_corpus_json = old_corpus_json["sentences"] # make it an array

# save it to another file (new_corpus.json)
with open("new_corpus.json", "w") as new_corpus:
    json.dump(new_corpus_json, new_corpus, ensure_ascii=False, indent=2)
