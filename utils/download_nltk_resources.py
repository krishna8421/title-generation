import nltk

# NLTK data path
nltk_data_path = "nltk_data"

# Set the NLTK data path to include our local directory
nltk.data.path.append(nltk_data_path)

# Function to check if NLTK data is present in the specified folder
def check_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        return False
    return True

# Function to download NLTK data to the specified folder
def download_nltk_data():
    # Check if NLTK data is present, if not, download it
    if not check_nltk_data():
        print("Downloading NLTK data...")
        # For sentence and word tokenization
        nltk.download("punkt", download_dir=nltk_data_path)
        # For stopwords list
        nltk.download("stopwords", download_dir=nltk_data_path)
