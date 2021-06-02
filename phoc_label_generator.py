"""
Module that generates 604 length PHOC vector as proposed in SPP-PHOCNet paper
Modified version from https://github.com/pinakinathc/phocnet_keras 
"""

import csv
import numpy as np

# Generates PHOC component corresponding to alphabets/digits

def generate_36(word):
    '''The vector is a binary and stands for:
    [0123456789abcdefghijklmnopqrstuvwxyz]
    '''
    vector_36 = [0 for i in range(36)]
    for char in word:
        if char.isdigit():
            vector_36[ord(char) - ord('0')] = 1
        elif char.isalpha():
            vector_36[10+ord(char) - ord('a')] = 1

    return vector_36

# Generates PHOC component corresponding to 50 most frequent bi-grams of English

def generate_50(word):
    bigram = ['th', 'he', 'in', 'er', 'an', 're', 'es', 'on', 'st', 'nt', 'en',
    'at', 'ed', 'nd', 'to', 'or', 'ea', 'ti', 'ar', 'te', 'ng', 'al',
    'it', 'as', 'is', 'ha', 'et', 'se', 'ou', 'of', 'le', 'sa', 've',
    'ro', 'ra', 'hi', 'ne', 'me', 'de', 'co', 'ta', 'ec', 'si', 'll',
    'so', 'na', 'li', 'la', 'el', 'ma']
    vector_50 = [0 for i in range(50)]
    for char in word:
        try:
            vector_50[bigram.index(char)] = 1
        except:
            continue

    return vector_50

# Input: A word(string) 
# Output: PHOC vector

def generate_phoc_vector(word):
    word = word.lower()
    vector = []
    L = len(word)
    for split in range(2, 6):
        parts = L//split
        for mul in range(split-1):
            vector += generate_36(word[mul*parts:mul*parts+parts])
        vector += generate_36(word[(split-1)*parts:L])
    # Append the most common 50 bigram text using L2 split
    vector += generate_50(word[0:L//2])
    vector += generate_50(word[L//2: L])
    return vector


# Input: A list of words(strings)
# Output: A dictionary of PHOC vectors in which the words serve as the key

def gen_phoc_label(word_list):
    label={}
    for word in word_list:
        label[word]=generate_phoc_vector(word)
    return label

# Input: A text file name that has a list of words(strings)
# Output: A dictionary of PHOC vectors in which the words serve as the key

def label_maker(word_txt):
    label={}
    with open(word_txt, "r") as file:
        for word_index, line in enumerate(file):
            word = line.split()[0]
            label[word]=gen_phoc_label(word)
    return label
    #write_s_file(s_matrix_csv, s_matrix, word_list)


