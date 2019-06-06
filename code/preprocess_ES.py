import sys
import re
import os
import numpy as np

from argparse import ArgumentParser

import xml.etree.ElementTree as ET

from collections import OrderedDict

import string

from nltk.corpus import stopwords

# Specify command line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("parent_path", help="The path of the Training (Eurosense XML) Data folder")

    return parser.parse_args()

class Processor:
    def __init__(self):
        """
        Processor class that Process the EuroSense XML file and creates the Word2Vec Training Dataset
        """

        print()
        # Create a folder to save the Training data (.npy) file in
        self.folder = "../resources/Processed_Training"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Path to BabelNet - WordNet mapping (.txt) file
        self.mapping_pth = os.path.join("../resources", "bn2wn_mapping.txt")
        # Path to (.txt) file where processed sentences will be saved
        self.processed_pth = os.path.join(self.folder, ("Training_sentences.txt"))
        # Path to (.npy) file where Training data will be saved
        self.train_data_pth = os.path.join(self.folder, ("Training_sentences.npy"))

    def open_xml(self, parent):
        """
        Opens the XML file and runs the Processing algorithm on it
        (See Section 2 "Processing" in report.pdf)
        """
        # Create a list of BabelNet sysnets that have WordNet offsets
        mapping = []
        with open(self.mapping_pth, 'r', encoding="utf8") as mp_f:
            mp_f_content = mp_f.readlines()
            for mp_line in mp_f_content:
                mapping.append(mp_line.split("\t")[0])
            # print(len(mapping))

        # Print which files will be used to build the Training data
        files = [parent]
        print()
        print("Training Dataset will be built from the following file(s) {:}...\n".format(files))

        # Open (Training_sentences.txt) to write processed sentences in
        no_sp = open(self.processed_pth , "w", encoding="utf8")
        # Initialize list of processed sentences and dictionary of "anchor:Lemma_Synset"
        sentences = []
        d = {}
        en_sentence = "-1"

        # Loop over all the dataset's files (1 in the case of EuroSense)
        for file in files:
            # Variable to count number of sentences processed and written to (.txt file)
            c = 0
            # Flags to check when all annotations for a sentence have been collected
            flag = True
            expecting_annotation = False
            # Parse the XML using iterparse as it is too big to load in memory
            for event, elem in ET.iterparse(parent, events=("start", "end")):

                if event == 'start' :
                    # Print number / "id" of sentence being processed
                    if elem.tag == 'sentence' :

                        prog = elem.get("id")
                        if (int(prog) < 40):
                            pass
                        else:
                            sys.stdout.write("Processing progress: %d   \r" % (int(prog)) )
                            sys.stdout.flush()

                        # Print the sentence at every step in the algorithm
                        # for the first 40 sentences, for debugging purposes
                        if (int(prog) < 40  and int(prog) > 0):
                            print("No punctuation: ", en_sentence)

                        # Sort the dictionary of anchor:lemma_synset by length of anchors
                        # To consider long annotation in overlapping annotations
                        d = OrderedDict(sorted(d.items(), key=lambda t: len(t[0].split()), reverse=True))

                        anchors_lengths = [len(k.split()) for k in d.keys()]


                        # Loop over anchors and consider the longest oens first
                        # Replace each anchor with lemma_synset
                        for idx, length in enumerate(anchors_lengths):
                            if length > 1:
                                # Add spaces before anchor to find only exact matces
                                long_anchor = (list(d.keys())[idx])
                                long_anchor = " " + long_anchor + " "
                                # Get lemma and Replace space between "lemma synset" with underscore
                                long_lemma_syn = (d[list(d.keys())[idx]])
                                long_lemma_syn = long_lemma_syn.replace(" ", "_")
                                # Finally Find and replace the anchor with lemma_synset
                                en_sentence = en_sentence.replace(long_anchor, " " + long_lemma_syn + " ")

                            elif length == 1:
                                # Add spaces before anchor to find only exact matces
                                anchor = (list(d.keys())[idx])
                                anchor = " " + anchor + " "
                                # Get lemma
                                lemma_syn = (d[list(d.keys())[idx]])
                                # Finally Find and replace the anchor with lemma_synset
                                en_sentence = en_sentence.replace(anchor, " " + lemma_syn + " ")

                        # Print the sentence at every step in the algorithm
                        # for the first 40 sentences, for debugging purposes
                        if (int(prog) < 40  and int(prog) > 0):
                            print("achor, lemma_synset dictionary: ", d)
                            print("processed:", en_sentence)
                            print()

                        # Write procesed sentence to file
                        if en_sentence != "-1":
                            with open (self.processed_pth, "a", encoding="utf8") as f:
                                c +=1
                                f.write(en_sentence)
                                f.write("\n")
                            f.close()

                    # Collect all English sentences
                    if elem.tag == 'text' and elem.text != None:
                        d = {}
                        if elem.get("lang") == "en":
                            if expecting_annotation == False:
                                expecting_annotation = True

                            flag = True
                            # Change all words in sentence to lower case
                            en_sentence = (elem.text).lower()

                            # Print the sentence at every step in the algorithm
                            # for the first 40 sentences, for debugging purpose
                            if (int(prog) < 40):
                                print(prog)
                                print("original:", en_sentence)

                            # Remove all punctation symbols from sentence
                            en_sentence = ''.join(" " if i in string.punctuation else i for i in en_sentence)
                            # Remove all digits from sentence
                            en_sentence = ''.join(i for i in en_sentence if not i.isdigit())
                            # Remove all duplicate spaces
                            en_sentence = re.sub(" +", " ", en_sentence)
                            # Add a space in the beginning and end of the sentence
                            # to find and replace correctly
                            en_sentence = (" ") + en_sentence + (" ")

                    # Get all English anchors, lemmas , and synsets
                    if elem.tag == 'annotation' and elem.get("lang") == "en":
                        if expecting_annotation == True:
                            expecting_annotation = False
                        if elem.text == None or elem.text not in mapping:
                            continue
                        else:
                            # Get anchor and change it to lower case
                            anchor = elem.get("anchor").lower()
                            # Remove all punctation symbols
                            anchor = ''.join(" " if i in string.punctuation else i for i in anchor)
                            # Remove all duplicate spaces
                            anchor = re.sub(" +", " ", anchor)

                            # Get lemma and change it to lower case
                            lemma = elem.get("lemma").lower()
                            # Remove all punctation symbols
                            lemma = ''.join(" " if i in string.punctuation else i for i in lemma)
                            # Remove all duplicate spaces
                            lemma = re.sub(" +", " ", lemma)

                            # Get synset
                            synset = elem.text
                            # Update the dictionary with the current
                            # anchor : lemma_synset
                            d.update({anchor: (lemma + "_" + synset)})

                elem.clear()

            print("Number of sentences in training_data txt file is ", c)

    def create_training(self):
        """
        Creates a list of (sentences) lists as the Training data for Word2Vec
        and saves it in an (.npy) file

        returns: Training data as list of lists
        """
        train = []
        print()
        print("Creating Word2Vec Training Data...")
        # Get list of English Stop-words from nltk.corpus
        stop_words = stopwords.words('english')
        # Remove punctiation symbols from Stop-words
        stop_words = [''.join(" " if i in string.punctuation else i for i in word) for word in stop_words]
        # Insert space before and after each word to be able to find and replace correctly
        stop_words = [" " + word + " " for word in stop_words]

        with open(self.processed_pth, 'r', encoding="utf8") as f:
            f_content = f.readlines()
            for line in f_content:
                # Replace Stop-words with blank space in original sentence
                for stop_word_found in [w for w in stop_words if w in line]:
                    line = line.replace(stop_word_found, " ")
                final_sentence = line.split()

                # Append final version of sentence to training data
                train.append(line.split())

        print("Done!")
        print("Lenth of Training Data is", len(train))
        # Save Training data as a (.npy) file for the train.py to load and train
        np.save(self.train_data_pth, train)
        print("Saved Training Data as NPY!")

        return train

if __name__ == '__main__':

    # Parse the command line arguments
    args = parse_args()

    # Instantiate an instance of the 'Procesor' class
    p = Processor()

    # Check if the dataset has been parsed and Training data has already been created
    if os.path.exists(p.processed_pth):
        print("Found Pre-processed and ready to go '.npy file' Training Dataset at {:}!".format(p.processed_pth))
        print("Just run 'train.py [embed_fname]' to train the model on it!")
    # Else, process the XML file and create the training data
    else:
        print("No Processed Dataset '.npy' file! Will parse & pre-process EuroSense XML file...")
        p.open_xml(args.parent_path)
        p.create_training()
