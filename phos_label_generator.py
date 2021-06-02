# Library imports

import csv
import numpy as np

# Input: CSV file name that has shape counts for each alphabet
# Output: Number of shapes/columns

def get_number_of_columns(csv_file):
    with open(csv_file) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        return len(next(reader))-1


# Input: CSV file name that has shape counts for each alphabet
# Output: A dictionary where alphabet is key mapped to its shape count vector(np-array)

def create_alphabet_dictionary(csv_file):
    alphabet_dict = dict()

    with open(csv_file) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)

        for index, line in enumerate(reader):
            alphabet_dict[line[0]] = index

    return alphabet_dict

alphabet_csv = "Alphabet.csv"

alphabet_dict = create_alphabet_dictionary(alphabet_csv)
csv_num_cols = get_number_of_columns(alphabet_csv)
numpy_csv = np.genfromtxt(alphabet_csv, dtype=int, delimiter=",")
numpy_csv=np.delete(numpy_csv,0,1)

# Input: A word segment(string)
# Output: A shape count vector for all alphabets in input word segment (np-array)

def word_vector(word):
    vector = np.zeros(csv_num_cols)
    for letter in word:
        letter_index = alphabet_dict[letter]
        vector += numpy_csv[letter_index]
    return vector

# Input: A word(string) 
# Output: PHOS vector

def generate_label(word):
    vector = word_vector(word)
    L = len(word)
    for split in range(2, 6):
        parts = L//split
        for mul in range(split-1):
            vector=np.concatenate((vector,word_vector(word[mul*parts:mul*parts+parts])),axis=0)
        vector=np.concatenate((vector,word_vector(word[(split-1)*parts:L])),axis=0)
    return vector

# Input: A list of words(strings)
# Output: A dictionary of PHOS vectors in which the words serve as the key

def gen_label(word_list):
    label={}
    for word in word_list:
        label[word]=generate_label(word)
    return label


# Input: A text file name that has a list of words(strings)
# Output: A dictionary of PHOS vectors in which the words serve as the key

def label_maker(word_txt):
    label={}
    with open(word_txt, "r") as file:
        for word_index, line in enumerate(file):
            word = line.split()[0]
            label[word]=generate_label(word)
    return label
    #write_s_file(s_matrix_csv, s_matrix, word_list)


