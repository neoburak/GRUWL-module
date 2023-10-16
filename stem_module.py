#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np 
import pandas as pd
import fasttext
import fasttext.util
import random
from itertools import permutations,combinations
from numpy.linalg import norm
import math
import random


# In[2]:


def loadModel(keyword):
    """   
    Parameters:
        keyword: string
            If keyword which is fasttext pre-trained model path or name of model such as "cc.tr.300.bin" is provided, 
            fasttext model can be used for further operations.
    Returns:
        object:
            If pre-trained model name is provided, returns fasttext model including word embedding vectors can be used for
            further operations.
            
    NOTE: if you have the model. It should be in the same folder with your python notebook.
    """   
    return fasttext.load_model(keyword)


# In[3]:


def getWordList(keyword):
    """   
    Parameters:
        keyword: string
            If keyword which has file path or name of file is provided, word list is  
            If matrix is not provided, it is the matrix for which pairwise cosine
            similarity between rows will be calculated. For example: getWordList("words2.txt") 
    
    Returns:
        array_like
            If word list is provided, returns an array of word list including each word in the file.
    """    
    word_list= pd.read_csv(keyword)
    word_list=np.array(word_list).flatten()
    return word_list


# In[4]:


def get_word_vectors(word_list, ft):
    """   
    Parameters:
        word_list: array_like
            If word list is provided in the form of numpy array, words in the list can be transformed word embbedings 
            (ie. word vectors) by fasttext model. 
        ft: object
            If the trained or pre-trained fasttext model is provided, word embeddings of word list can be prepared.
    
    Returns:
        array_like
            If word list is provided, returns word vectors of the list in the form of numpy array.
    """    
    return np.array(list(map(ft.get_word_vector, word_list)))


# In[5]:


def cos_similarity_function(vector_or_matrix_1, matrix=None):
    """
    Calculate cosine similarity between a vector and all rows of a matrix
    or pairwise cosine similarity between all rows of a matrix.
    
    Parameters:
        vector_or_matrix_1: array_like
            If matrix is provided, it is the vector for which cosine similarity
            will be calculated against all rows of the matrix.
            If matrix is not provided, it is the matrix for which pairwise cosine
            similarity between rows will be calculated.
        
        matrix: array_like, optional
            The matrix against which cosine similarity will be calculated.
            If not provided, pairwise cosine similarity will be calculated
            between all rows of vector_or_matrix_1.
    
    Returns:
        array_like
            If matrix is provided, returns an array of cosine similarities between
            the vector and each row of the matrix.
            If matrix is not provided, returns a matrix of pairwise cosine similarities
            between all rows of the vector_or_matrix_1.
    """
    if matrix is None:
        matrix = vector_or_matrix_1
        return np.dot(matrix, matrix.T) / (np.linalg.norm(matrix, axis=1)[:, np.newaxis] * np.linalg.norm(matrix, axis=1))
    else:
        vector = vector_or_matrix_1
        return np.dot(vector, matrix.T) / (np.linalg.norm(vector) * np.linalg.norm(matrix, axis=1))


# In[37]:


if __name__ == "__main__":

    ft = loadModel(r'cc.tr.300.bin') 
    word_list= getWordList("words2.txt") 


# In[38]:


if __name__ == "__main__":
    vectors = get_word_vectors(word_list, ft)
    word_distances = cos_similarity_function(vectors)


# In[6]:


def cosine_similarity(vector1, vector2):
    """   
    Parameters:
        vector1: array_like
            If word vector is provided in numpy array form, cosine similarity can be estimated with regard to other word vector.         vector2: array_like
        vector2: array_like
            If word vector is provided in numpy array form, cosine similarity can be estimated with regard to other word vector.
    Returns:
        float
            If both of word vectors provided, returns cosine similarity result.
    """ 
    
    
    dot_product = np.dot(vector1, vector2)
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)
    return dot_product / (norm_1 * norm_2)


# In[7]:


def generate_one_list(model_name,word_distances,word_list,size_of_list=12,cosine_similarity_value=0.30):
    """   
    Parameters:
        model_name: object
            If pre-trained model name is provided, model including word embedding vectors can be used to
            generate given size and cosine similarity value.
            
        word_distances: array_like
            If word distances which was generated with "cos_similarity_function" function is provided in numpy array form, 
            word distances can be used to estimated the furthest item with regard to cosine similarity value parameter.
        word_list: array_like
            If word list is provided in numpy array form, dissimlar word list is generated from the given word list.
        size_of_list: int, optinal
            If size of list is provided, a dissimilar word list is generated with number of items which is assigned.
        cosine_similartiy_value: float, optinal
            If cosine similarity value is provided cosine similarity value of words will not be higher than assigned
            cosine similarity value. If any cosine similary value is  in the dissimilar word list
    Returns:
        array_like
            If distances between words, word list in the form of numpy array and provided, returns dissimilar list determined 
            number  of  items with regard to cosine similarity value.
        
    """  
    temp1=[]
    temp2=[]
    u=[]
    temp1.append(word_list[np.argpartition(word_distances[random.randint(0,len(word_list))],
                                                                random.randint(0,5))[random.randint(0,5)]])
   
    while True:
        if (len(u)==size_of_list):
            break
        a= random.randint(0,len(word_list))
        b= random.randint(0,5)
        
        try:
            index=np.argpartition(word_distances[a], b)[b]
        except:
            print("index hatası")
        else:
            index=np.argpartition(word_distances[a], b)[b]   
            temp1.append(word_list[index])
            for a,b in combinations(temp1,2):
                if(cosine_similarity_value<cosine_similarity(model_name[a],model_name[b])):
                    temp2.append(a)
            u=list(set(temp1) - set(temp2))
    return u


# In[57]:


if __name__ == "__main__":
    res= generate_one_list(ft,word_distances,word_list)
    res=[]
    i=0
    while i<12:
        res.append(generate_one_list(ft,word_distances,word_list,20))
        i=i+1
    print(res)


# In[12]:


# delete repeated items
def unique_values_in_list_of_lists(word_list):
    """   
    Parameters:
        word_list: array_like
            If word list is provided in numpy array form, dissimilar word list is generated from the given word list.

    Returns:
        array_like
            If distances between words, word list in the form of numpy array and provided, returns dissimilar list determined 
            number  of  items with regard to cosine similarity value.
        
    """  
    result=[]
    result.append(list(set(x for l in word_list for x in l)))
    return result


# In[13]:


# finally crate lists contain 10 items
if __name__ == "__main__":

    x=[]
    w=[]
    x= unique_values_in_list_of_lists(res)[0]
    #x1= x[:50]

    w=[x[i:i+20] for i in range(0, len(x), 10)]
    w


# In[14]:


#get nearest
def generate_similar_words(word_list,word_distances,size_of_list=12,print_values=False):

    """   
    Parameters:
        word_list: array_like
            If word list is provided in numpy array form, similar word list is generated from the given word list.
        word_distances: array_like
            If word distances which was generated with "cos_similarity_function" function is provided in numpy array form, 
            word distances can be used to estimated the nearest items.
        size_of_list: int, optinal
            If size of list is provided, a similar word list is generated with number of items which is assigned.
        print_values: bool, optinal
            If print_values paramater is set true, only values of dictonary is returned 
    Returns:
        array_like
            If print value is assigned true and word list, distances between words 
            in the form of numpy array and provided, returns similar list determined number  of  items in form of list.
        dictionary_like
            If print value is assigned false and word list, distances between words 
            in the form of numpy array and provided, returns similar list determined number  of  items in form of dictionary.            
            
    """  
    
    indices = np.apply_along_axis(lambda x: np.argsort(x, axis=0)[-size_of_list:-1], 1, word_distances)
    distant_words = np.apply_along_axis(lambda x:word_list[x], 1, indices)
    dist_words_dict = dict(zip(word_list, distant_words.tolist()))
    if print_values==True:
        result= list(dist_words_dict.values())
        return [item for sublist in result for item in sublist]
    else:
        return list(dist_words_dict.values())


# In[15]:


if __name__ == "__main__":
    print(generate_similar_words(word_list,word_distances,12))


# ## Diagnosis:

# In[50]:


def show_diagnosis(model_name,word_list):
    temp=[]
    """   
    Parameters:
        word_list: array_like
            If word list is provided in numpy array form, cosine similarity of combination of items of given list.
    Returns:
        array_like
            If word list, distances between words in the form of numpy array and provided, returns similar list determined 
            number  of  items.
    NOTE: before using show_diagnosis function you must 
    """      
    for a,b in combinations(word_list,2):
      temp.append((a,b,cosine_similarity(model_name[a],model_name[b])))
    return temp


# ## Sonuçları Görüntüleme ve Yazdırma

# In[ ]:


def making_list_and_print(word_list, num_of_lists=20, write_lists= False):
    """   
    Parameters:
        word_list: array_like
            If word list is provided in numpy array form, cosine similarity of combination of items of given list.
        num_of_lists: int, optional
            If number of lists is provided, it groups with regard to items given number of lists.

    Returns:
        array_like
            If word list, and number of lists are provided, it returns grouped lists.
    NOTE: before using show_diagnosis function you must 
    """       
    res = []
    for i in range(0, len(word_list), num_of_lists):
        temp = word_list[i:i+num_of_lists]
        res.append(temp)
    if write_lists== True:
        with open("result_list.txt", "w") as output:
            output.write(str(gruplu_listeler))
    else:   
        return res


# In[ ]:


# Örnek kullanım:
if __name__ == "__main__":
    item_listesi = ["item" + str(i) for i in range(1, 161)]

    gruplu_listeler = making_list_and_print(word_list)

    # Her 20'li grup için ayrı ayrı listeleri ekrana yazdırma
    for index, grup in enumerate(gruplu_listeler):
        print(f"Grup {index+1}: {grup}")

