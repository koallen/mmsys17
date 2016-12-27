import jieba

class SimpleClassifier:

    def classify(self, sentence, correct_sentiment, dictionary, inverse):
        word_in_sentence = jieba.lcut(sentence)
        total_score = 0
        i = 0 # word index
        s = -1 # last sentiment word index
        for word in word_in_sentence:
            # print("Word is " + word + " at " + str(i))
            if word in dictionary and word not in inverse:
                # print(str(i) + " in dict")
                partial_score = 1
                # print("Scanning " + str(s+1) + " to " + str(i-1))
                for p_word in word_in_sentence[s+1:i]:
                    if p_word in inverse:
                        # print("Inverting sentiment score")
                        partial_score *= -1
                # print("Scan complete")
                if dictionary[word] == 'p':
                    total_score += partial_score
                elif dictionary[word] == 'n':
                    total_score -= partial_score
                s = i # set last sentiment word index as current index
            i += 1 # increment index
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
            return (True, correct_sentiment, sentence_sentiment)
        else:
            return (False, correct_sentiment, sentence_sentiment)
