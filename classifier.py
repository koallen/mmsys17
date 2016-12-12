import jieba

class SentimentClassifier:

    def simple(self, sentence, correct_sentiment, dictionary):
        word_in_sentence = jieba.cut(sentence)
        total_score = 0
        for word in word_in_sentence:
            if word in dictionary:
                if dictionary[word] == 'p':
                    total_score += 1
                elif dictionary[word] == 'n':
                    total_score -= 1
        if total_score > 0:
            sentence_sentiment = 'p'
            # print("Sentence is positive")
        elif total_score < 0:
            sentence_sentiment = 'n'
            # print("Sentence is negative")
        else:
            sentence_sentiment = 'z'
            # print("Sentence is neutral")
        if correct_sentiment == sentence_sentiment:
            return True
        else:
            return False
