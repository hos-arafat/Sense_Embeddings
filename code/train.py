import os
import re
import numpy as np
from argparse import ArgumentParser


import multiprocessing
from time import time
from gensim.models import Word2Vec
import logging

from preprocess_ES import Processor
from test import Tester

# Specify command line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("embed_fname", help="The name of the file in which to save the sense embeddings")

    return parser.parse_args()

class Model:
    def __init__(self):
        """
        Model class that implements Gensim's Word2Vec model
        """

        p = Processor()

        # Get path to (.npy) Training data from Processor object
        self.train_data_pth = p.train_data_pth

        # Compile Regular Expression that detects if string "*_bn:01234567n" is present
        self.syn_reg = re.compile(r'[A-Za-z]_bn\:[0-9]{8}[a-z]')



    def train_word2vec(self, w, embed_size, lr):
        """
        Trains CBOW Word2Vec model for one experiment with the best value of the hyper-parameters
        found after Gird Search (See Appendix A.2 in report.pdf)
        """
        # Specify formatting of logging the training progress
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # Load (.npy) Training data, exit otherwise
        if os.path.exists(self.train_data_pth):
            training_data = np.load(self.train_data_pth)
            print("Loaded Training Data NPY !")
        else:
            print("Unable to Load Training Data NPY...Please run 'preprocess_ES.py [parent_path]'")
            return

        cores = multiprocessing.cpu_count() # Count the number of cores in a computer

        print()
        print("Training Word2Vec Model....")
        print("Number of cores is ", cores)

        # Initialize Word2Vec model
        w2v_model = Word2Vec(min_count=5, window=w, size=embed_size, sample=1e-3,
                     alpha=lr,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)
        # Claculate time taken to load vocab into model
        t = time()

        w2v_model.build_vocab(training_data, progress_per=10000)
        print(w2v_model)
        # words = list(w2v_model.wv.vocab)
        # print(words)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        # Claculate time taken to train the model once, for 30 epochs
        t = time()

        w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

        return w2v_model


    def is_label(self, word):
        """
        Function that returns true whenever Regular Expression is met
        """
        if bool(re.search(self.syn_reg, word)):
            return True
        else:
            return False

    def save_embeddings(self, w2v_model, path_to_save):
        """
        Save ONLY sense embeddings to (.vec) file
        """
        X = w2v_model[w2v_model.wv.vocab]

        print()
        print("Will save embeddings.vec to ", path_to_save)
        print("Embeddings Matrix type is ", type(X))
        print("Embeddings Matrix size ", X.shape)

        # Loop over the model's vocab and extract only senses by looking
        # for the sub-string "*_bn:01234567*" in them
        sense_embeds = []
        all_tokens = w2v_model.wv.vocab
        for token in all_tokens:
            if(self.is_label(token)):
                sense_embeds.append(token)

        print("Number of Sense embeddings ", len(sense_embeds))

        # Write sense embeddings to file
        with open(path_to_save, "w") as f:
            f.write(str(len(sense_embeds)) + " " + str(X.shape[1]))
            f.write("\n")
        f.close()
        for key in sense_embeds:
            with open(path_to_save, "a") as f:
                f.write(key + " ")
                for element in w2v_model[key]:
                    f.write(str(element) + " ")
                f.write("\n")
            f.close()

        print("Successfully Saved embeddings!")


if __name__ == '__main__':

    args = parse_args()

    p = Processor()
    m = Model()

    # If the processed training sentences exist, train the model and save it
    if os.path.exists(p.processed_pth):
        print("Found Pre-pocessed Dataset! Training..")
        m.save_embeddings(m.train_word2vec(10, 350, 0.02), args.embed_fname)
    # Else, instruct user to run the "preprocess" code first
    else:
        print("No Processed Dataset! Please run 'preprocess_ES.py [parent_path]'")
        print("where:")
        print("parent_path: Path to the eurosense.v1.0.high-precision.xml file")
