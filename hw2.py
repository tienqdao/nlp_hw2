# HW2.py by Tien Dao and Dat Tran - CAP 4641

from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.naive_bayes
import random
from collections import Counter
import math
import re

# Global variables to store n-gram models - Tien:
_ngram_model = {
    "trained": False,
    "unigram": Counter(),
    "bigram": Counter(),
    "trigram": Counter(),
    "bigram_context": Counter(),
    "trigram_context": Counter(),
    "vocab": set(),
    "V": 0,
    "total_tokens": 0,
    "alpha": 0.5
}

# Helper functions for Problem 1 - Tien:
def _tokenize(text):
    """
    Simple tokenizer:
    - lowercase
    - keep letters/numbers/apostrophes
    """
    return re.findall(r"[a-z0-9']+", text.lower())

def _sentence_score(sentence):
    """
    Returns average log probability per token.
    Lower = less likely under the trained language model.
    """
    global _ngram_model

    unigram = _ngram_model["unigram"]
    bigram = _ngram_model["bigram"]
    trigram = _ngram_model["trigram"]
    bigram_context = _ngram_model["bigram_context"]
    trigram_context = _ngram_model["trigram_context"]
    vocab = _ngram_model["vocab"]
    V = _ngram_model["V"]
    total_tokens = _ngram_model["total_tokens"]
    alpha = _ngram_model["alpha"]

    tokens = _tokenize(sentence)

    # Map unseen words to <UNK>
    tokens = [tok if tok in vocab else "<UNK>" for tok in tokens]

    # Add sentence boundary markers
    seq = ["<s>", "<s>"] + tokens + ["</s>"]

    total_log_prob = 0.0
    count = 0

    # Interpolated trigram model
    lam3 = 0.7
    lam2 = 0.2
    lam1 = 0.1

    for i in range(2, len(seq)):
        w1, w2, w3 = seq[i - 2], seq[i - 1], seq[i]

        # Add-alpha smoothed trigram
        p3 = (trigram[(w1, w2, w3)] + alpha) / (trigram_context[(w1, w2)] + alpha * V)

        # Add-alpha smoothed bigram
        p2 = (bigram[(w2, w3)] + alpha) / (bigram_context[w2] + alpha * V)

        # Add-alpha smoothed unigram
        p1 = (unigram[w3] + alpha) / (total_tokens + alpha * V)

        p = lam3 * p3 + lam2 * p2 + lam1 * p1
        total_log_prob += math.log(p)
        count += 1

    return total_log_prob / max(count, 1)


"""
trainFile: a text file, where each line is arbitratry human-generated text
Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
"""
def calcNGrams_train(trainFile):
    global _ngram_model

    unigram = Counter()
    bigram = Counter()
    trigram = Counter()
    bigram_context = Counter()
    trigram_context = Counter()
    vocab = set()
    total_tokens = 0

    # First pass: collect raw token counts
    with open(trainFile, "r", encoding="utf-8") as f:
        for line in f:
            tokens = _tokenize(line)
            if not tokens:
                continue

            seq = ["<s>", "<s>"] + tokens + ["</s>"]

            for i in range(2, len(seq)):
                w1, w2, w3 = seq[i - 2], seq[i - 1], seq[i]

                unigram[w3] += 1
                bigram[(w2, w3)] += 1
                trigram[(w1, w2, w3)] += 1
                bigram_context[w2] += 1
                trigram_context[(w1, w2)] += 1

                vocab.add(w3)
                total_tokens += 1

    # Add an UNK token to handle unseen words at test time
    vocab.add("<UNK>")

    _ngram_model = {
        "trained": True,
        "unigram": unigram,
        "bigram": bigram,
        "trigram": trigram,
        "bigram_context": bigram_context,
        "trigram_context": trigram_context,
        "vocab": vocab,
        "V": len(vocab),
        "total_tokens": total_tokens,
        "alpha": 0.5
    }

    # do not return anything
    pass

"""
sentences: A list of single sentences. All but one of these consists of entirely random words.
Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is random.
"""
def calcNGrams_test(sentences):
    global _ngram_model

    # Score every sentence; the random one should have the lowest score
    scores = [_sentence_score(s) for s in sentences]

    random_index = min(range(len(scores)), key=lambda i: scores[i])
    return random_index

"""
trainFile: A jsonlist file, where each line is a json object. Each object contains:
	"review": A string which is the review of a movie
	"sentiment": A Boolean value, True if it was a positive review, False if it was a negative review.
"""
def calcSentiment_train(trainFile):
	pass #don't return anything from this function!

"""
review: A string which is a review of a movie
Return a boolean which is the predicted sentiment of the review.
Must run in under 120 seconds, and must use Naive Bayes
"""
def calcSentiment_test(review):
	return random.choice([True, False])
