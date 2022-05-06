import logging
import numpy as np
import nltk
import tqdm
import string

from tqdm import tqdm
from logging import info as info
from logging import warning as warning
from logging import exception as exception
from logging import error as error


logging.basicConfig(filename='tokenization.log', level=logging.INFO)
info = print

class Tokenizer:
     
    
    def __init__(
        self,
        qgrams = None, 
        is_char_tokenization = None, 
        clean = None
    ):
        
        self.qgrams = qgrams
        self.is_char_tokenization = is_char_tokenization
        self.clean = clean
        
        info("Tokenization initialized with.. ")
        info("- Q-gramms: ", self.qgrams)
        info("- Char-Tokenization: ", self.is_char_tokenization)
        info("- Text cleanning process: ", self.clean)
        
    def process(self, data):
        
        # if isinstance(data, list):
        # elif isinstance(data, pd.DataFrame):
        # elif isinstance(data, np.array):
            
        self.data_size = len(data)
        self.data = np.array(data, dtype = object)
        self.tokenized_data = np.empty([self.data_size], dtype = object)
        
        info("\nProcessing strarts.. ")
        info("- Data size: ", self.data_size)
        
        # self.data_mapping = np.array(input_strings, dtype=object)
        
        for i in tqdm(range(0, self.data_size), desc="Processing.."):
            if self.clean is not None:
                string = self.clean(self.data[i])
            else:
                string = self.data[i]
            # info(string)
            if self.is_char_tokenization:
                self.tokenized_data[i] = set(nltk.ngrams(string, n = self.qgrams))
            else:
                if len(nltk.word_tokenize(string)) > self.qgrams:
                    self.tokenized_data[i] = set(nltk.ngrams(nltk.word_tokenize(string), n = self.qgrams))
                else:
                    self.tokenized_data[i] = set(nltk.ngrams(nltk.word_tokenize(string), n = len(nltk.word_tokenize(string))))
            # info(self.tokenized_data[i])

        return self.tokenized_data


def clean(s):
    
    new_s = " ".join(s)

    # Lower letters 
    new_s = new_s.lower()
    
    # Remove unwanted chars 
    new_s = new_s.replace("\n", " ").replace("/z", " ")
    
    # Remove pancutation     
    new_s = new_s.translate(str.maketrans('', '', string.punctuation))
    
    return str(new_s)