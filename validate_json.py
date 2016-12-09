from jsonschema import validate
import json

schema = {
    "content": "a sentence",
    "influences": [],
    "id": "numbers",
    "sentiment": "p",
    "contain_words": [
        {
            "privative": False,
            "word": "a word",
            "adverb_of_degree": False,
            "word_semantic": "a semantic",
            "word_sentiment": "a sentiment"
        }
    ]
}

with open("new_corpus.json", "r") as corpus:
    corpus_json = json.load(corpus)
    try:
        validate(corpus_json, schema)
        print("Validation successful")
    except Exception as e:
        print(e)
        print("Validation failed")
