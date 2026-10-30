"""Fallbacks for the NLTK data MTO touches (stopwords, punkt, POS tagger) when it cannot be downloaded.

Import this before anything else imports nltk. If the real data is installed nothing is changed.
"""
import nltk
from nltk.tokenize import NLTKWordTokenizer


def _have(path):
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


class _Stopwords:
    def words(self, lang="english"):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return sorted(ENGLISH_STOP_WORDS)


if not _have("corpora/stopwords"):
    import nltk.corpus
    nltk.corpus.stopwords = _Stopwords()

if not (_have("tokenizers/punkt") or _have("tokenizers/punkt_tab")):
    import nltk.tokenize
    _word = NLTKWordTokenizer()

    def _sent_tokenize(text, language="english"):
        import re
        return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]

    nltk.tokenize.word_tokenize = lambda text, language="english", preserve_line=False: _word.tokenize(text)
    nltk.tokenize.sent_tokenize = _sent_tokenize
    nltk.word_tokenize, nltk.sent_tokenize = nltk.tokenize.word_tokenize, _sent_tokenize

if not (_have("taggers/averaged_perceptron_tagger") or _have("taggers/averaged_perceptron_tagger_eng")):
    def _pos_tag(words, *a, **k):
        # crude: -ly words are adverbs, capitalised non-initial words proper nouns
        return [(w, "RB" if w.lower().endswith("ly") else "NNP" if i and w[:1].isupper() else "NN")
                for i, w in enumerate(words)]
    nltk.pos_tag = _pos_tag
