import string

def preprocess(row):

    paper_str = " ".join(row)

    # Lower letters 
    paper_str = paper_str.lower()
    
    # Remove unwanted chars 
    paper_str = paper_str.replace("\n", " ").replace("/z", " ")
    
    # Remove pancutation     
    paper_str = paper_str.translate(str.maketrans('', '', string.punctuation))
    
    return str(paper_str)