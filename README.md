# Auto-Complete Bot

A natural language processing (NLP) project that implements an intelligent text auto-completion system using N-gram language models with add-k smoothing. The bot predicts the next word based on previous tokens by analyzing patterns in Twitter text data.

## Features

- **N-gram Language Model**: Implements 1-gram through 5-gram models for word prediction
- **Add-k Smoothing**: Uses Laplace smoothing to handle unseen word combinations
- **Out-of-Vocabulary (OOV) Handling**: Manages unknown words using a vocabulary threshold
- **Data Preprocessing**: Cleans and tokenizes text data for optimal model performance
- **Multiple Suggestions**: Returns predictions from different N-gram models with probability scores
- **Prefix Filtering**: Optional filtering to suggest words starting with specific characters

## Project Structure

```
Auto-Complete-Bot/
├── data/
│   └── en_US.twitter.txt          # Training data (Twitter corpus)
├── src/
│   ├── Core/
│   │   ├── DataSplit.py           # Train/test data splitting
│   │   ├── Model.py               # N-gram model and prediction logic
│   │   ├── NLPClass.py            # Vocabulary and OOV handling
│   │   ├── Preprocess.py          # Text preprocessing and tokenization
│   │   └── __init__.py
│   ├── Pipeline.py                # Main pipeline orchestration
│   └── __init__.py
├── main.py                        # Entry point and usage example
├── requirements                   # Project dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Auto-Complete-Bot.git
cd Auto-Complete-Bot
```

2. Install dependencies:
```bash
pip install -r requirements
```

3. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
```

## Usage

### Basic Example

```python
from src.Pipeline import Pipeline

# Initialize the pipeline (loads and trains on default Twitter data)
pipeline = Pipeline()

# Get word suggestions for a given context
suggestions = pipeline.get_suggestions("i was about")
print(suggestions)
```

### Advanced Usage

```python
# Use custom data file
pipeline = Pipeline(path='path/to/your/data.txt')

# Get suggestions with custom smoothing parameter
suggestions = pipeline.get_suggestions("i was about", k=1.5)

# Filter suggestions to words starting with specific prefix
suggestions = pipeline.get_suggestions("i was about", k=1.0, start_with="to")
```

## How It Works

### 1. **Preprocessing** (`Preprocess.py`)
- Reads text data from file
- Removes non-alphabetic characters
- Tokenizes sentences into words
- Converts text to lowercase

### 2. **Data Splitting** (`DataSplit.py`)
- Splits data into training (80%) and testing (20%) sets
- Uses stratified random sampling

### 3. **Vocabulary Building** (`NLPClass.py`)
- Counts word frequencies in training data
- Creates vocabulary from words appearing ≥50 times (threshold)
- Replaces rare words with `<unk>` token

### 4. **N-gram Model** (`Model.py`)
- Builds N-gram counts (1-gram to 5-gram)
- Calculates probabilities using add-k smoothing:
  ```
  P(w|context) = (count(context, w) + k) / (count(context) + k * |V|)
  ```
- Returns word with highest probability for each N-gram model

### 5. **Pipeline** (`Pipeline.py`)
- Orchestrates the entire workflow
- Provides simple interface for predictions

## Model Details

### N-gram Models
The system builds multiple N-gram models:
- **Unigram**: Predicts based on word frequency alone
- **Bigram**: Considers the previous 1 word
- **Trigram**: Considers the previous 2 words
- **4-gram**: Considers the previous 3 words
- **5-gram**: Considers the previous 4 words

### Smoothing
Add-k smoothing (default k=1.0) prevents zero probabilities for unseen N-grams and provides better generalization.

## Output Format

The `get_suggestions()` method returns a list of tuples, one for each N-gram model:
```python
[
    (suggested_word_1gram, probability_1gram),
    (suggested_word_2gram, probability_2gram),
    (suggested_word_3gram, probability_3gram),
    (suggested_word_4gram, probability_4gram)
]
```

## Dependencies

- `nltk` - Natural Language Toolkit for tokenization
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Train/test splitting

## Example Output

```python
pipeline = Pipeline()
suggestions = pipeline.get_suggestions("i was about")
# Output: [('to', 0.023), ('to', 0.156), ('to', 0.342), ('to', 0.289)]
```

## Limitations

- Requires sufficient training data for accurate predictions
- Performance depends on vocabulary threshold settings
- Limited to English language (based on training data)
- Memory intensive for very large corpora

## Future Improvements

- [ ] Add support for neural language models (LSTM/Transformer)
- [ ] Implement beam search for better predictions
- [ ] Add caching for faster repeated queries
- [ ] Support for multiple languages
- [ ] Web interface for interactive usage

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created as an NLP learning project to understand language modeling and text prediction.
