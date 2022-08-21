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

def clean(s):
    
    new_s = " ".join(s)

    # Lower letters 
    new_s = new_s.lower()
    
    # Remove unwanted chars 
    new_s = new_s.replace("\n", " ").replace("/z", " ")
    
    # Remove pancutation     
    new_s = new_s.translate(str.maketrans('', '', string.punctuation))
    
    return str(new_s)