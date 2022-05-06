import string

def preprocess(s):

    new_s = " ".join(s)

    # Lower letters 
    new_s = new_s.lower()
    
    # Remove unwanted chars 
    new_s = new_s.replace("\n", " ").replace("/z", " ")
    
    # Remove pancutation     
    new_s = new_s.translate(str.maketrans('', '', string.punctuation))
    
    return str(new_s)