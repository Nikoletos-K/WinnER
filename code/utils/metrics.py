'''
@author: Konstantinos Nikoletos, 2021
'''
import math

def rbo(list1, list2, p=0.9):

    # tail recursive helper function 
    def helper(ret, i, d):    
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2))/i
        term = math.pow(p, i) * a_d
        
        if d == i:
           return ret + term
    
        return helper(ret + term, i + 1, d)

    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)

    return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)


def rbo(list1, list2, p=0.9):

    # tail recursive helper function 
    def helper(ret, i, d):    
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2))/i
        term = math.pow(p, i) * a_d
        
        if d == i:
           return ret + term
    
        return helper(ret + term, i + 1, d)

    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)

    return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)

def spearman_footrule_distance(s,t):
    """
    Computes the Spearman footrule distance between two full lists of ranks:
        F(s,t) = sum[ |s(i) - t(i)| ]/S,
    the normalized sum over all elements in a set of the absolute difference between
    the rank according to s and t.  As defined, 0 <= F(s,t) <= 1.
    S is a normalizer which is equal to 0.5*len(s)^2 for even length ranklists and
    0.5*(len(s)^2 - 1) for odd length ranklists.
    If s,t are *not* full, this function should not be used. s,t should be array-like
    (lists are OK).
    """
    # check that size of intersection = size of s,t?
    assert len(s) == len(t)
    sdist = sum(abs(s - t))
    # c will be 1 for odd length lists and 0 for even ones
    c = len(s) % 2
    normalizer = 0.5*(len(s)**2 - c)
    
    return sdist/normalizer