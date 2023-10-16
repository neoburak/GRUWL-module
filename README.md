# STEM MODULE

# Word Embedding Operations with FastText

This repository contains Python code for working with word embeddings using the FastText library. It includes functions for loading FastText models, calculating cosine similarity between words, generating similar and dissimilar word lists, and more.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- FastText library
- Numpy
- Pandas

## Getting Started

To get started, follow these steps:

1. Clone this repository to your local machine: git clone https://github.com/neoburak/Memory-Thesis.git
2.  Install the required dependencies:
a.  pip install fasttext
b.  pip install numpy
c.  pip install pandas
3. Run the code using Python.

## Usage

Here are some examples of how to use the functions provided in this code:

### Loading a FastText Model

```python
ft = loadModel(r'cc.tr.300.bin')

### Calculating Cosine Similarity

cosine_sim = cosine_similarity(vector1, vector2)

### Generating Dissimilar Word Lists

dissimilar_words = generate_one_list(ft, word_distances, word_list)

### Generating Similar Word Lists

similar_words = generate_similar_words(word_list, word_distances, 12)
```


## Additional Information
show_diagnosis function can be used to diagnose the cosine similarity between different words in the word list.
making_list_and_print function groups words into lists and can print them or write them to a file.

## License
fastText is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- FastText
- Numpy
- Pandas
- Barbaros YET
- Burak BÜYÜKYAPRAK
- GitHub: https://github.com/neoburak

Happy coding!

