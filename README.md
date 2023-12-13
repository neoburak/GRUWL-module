# STEM MODULE

# Word Embedding Operations with FastText for Episodic Memory Studies

This repository contains Python code for working with word embeddings using the FastText library. It includes functions for loading FastText models, calculating cosine similarity between words, generating similar and dissimilar word lists, and more.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- FastText library
- Numpy
- Pandas

You can download the Turkish pre-trained Word2Vec model from here: https://github.com/akoksal/Turkish-Word2Vec
You can download the Turkish pre-trained fastText model from here: https://fasttext.cc/docs/en/crawl-vectors.html


## Getting Started

To get started, follow these steps:

1. Clone this repository to your local machine: git clone https://github.com/neoburak/Memory-Thesis.git
2.  Install the required dependencies:
a.  pip install fasttext
b.  pip install numpy
c.  pip install pandas
d.  pip install gensim

3. Run the code using Python.

## Usage

Here are some examples of how to use the functions provided in this code:

### Loading a FastText Model

```python
### Import module
import stem_module as sm
 
### Import fasttext model
ft = sm.loadModel(r'cc.tr.300.bin')

### Calculating Cosine Similarity

cosine_sim = sm.cosine_similarity(vector1, vector2)

### Generating Dissimilar Word Lists

dissimilar_words = sm.generate_one_list(ft, word_distances, word_list)

### Generating Similar Word Lists

similar_words = sm.generate_similar_words(word_list, word_distances, 12)

### Compare Word2Vec Corpus and Word List

word2vec_list= sm.word2vec_checkword_list(word_list,word2vec_model)

```


## Additional Information
show_diagnosis function can be used to diagnose the cosine similarity between different words in the word list.
making_list_and_print function groups words into lists and can print them or write them to a file.

## License
fastText is licensed under the MIT License

## Acknowledgments
- FastText
- Numpy
- Pandas
## Author
- Barbaros YET
- GitHub: https://github.com/byet
- Burak BÜYÜKYAPRAK
- GitHub: https://github.com/neoburak

Happy coding!

