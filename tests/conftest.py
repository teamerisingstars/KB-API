import pytest
import nltk

@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    for corpus in ["punkt_tab", "punkt", "averaged_perceptron_tagger_eng",
                   "averaged_perceptron_tagger", "wordnet", "stopwords", "omw-1.4"]:
        nltk.download(corpus, quiet=True)
