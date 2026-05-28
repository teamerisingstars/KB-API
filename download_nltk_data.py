import nltk
import os
import zipfile

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

# Some NLTK versions / Python versions download zip files but don't auto-extract.
# Extract any remaining zips in the corpora directory so nltk.data.find() works.
for data_dir in nltk.data.path:
    corpora_dir = os.path.join(data_dir, "corpora")
    if not os.path.isdir(corpora_dir):
        continue
    for fname in os.listdir(corpora_dir):
        if not fname.endswith(".zip"):
            continue
        extracted_name = fname[:-4]
        if not os.path.exists(os.path.join(corpora_dir, extracted_name)):
            with zipfile.ZipFile(os.path.join(corpora_dir, fname)) as zf:
                zf.extractall(corpora_dir)

print("NLTK data downloaded and extracted.")
