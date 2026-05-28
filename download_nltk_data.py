import nltk

REQUIRED = [
    "punkt_tab",
    "punkt",
    "averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger",
    "wordnet",
    "stopwords",
    "omw-1.4",
]

for corpus in REQUIRED:
    nltk.download(corpus, quiet=True)

print("NLTK data downloaded.")
