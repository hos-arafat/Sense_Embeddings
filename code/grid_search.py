import os
from argparse import ArgumentParser

from preprocess_ES import Processor
from test import Tester
from train import Model

import operator

# Specify command line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("test_f_path", help="Path to the Testing (Secret) '.tab' file")
    return parser.parse_args()

def grid_search(p, m, t, test_path):
    """
    Implements Grid Search Algorithm (See section 3.2 and Appendix A.2 of report.pdf)
    """

    # Specify folder to save embeddings obtained as result of training
    folder = "../resources/Grid_Search_Embeddings"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Dictionary that has the parameter configuration: Spearman Score
    results = {}

    # Values of hyper parammeters to go through
    windows = [5, 7, 10]
    embedding_sizes = [300, 350, 400]
    l_rates = [0.02, 0.03, 0.04]
    # Train the model, save it, and test it on Word Similarity
    for w in windows:
        for embed_size in embedding_sizes:
            for lr in l_rates:
                print("Saving embeddings to ", str(w) + "_" + str(embed_size) + "_" + str(lr) + ".vec")
                path_to_save = os.path.join(folder, (str(w) + "_" + str(embed_size) + "_" + str(lr) + ".vec"))

                w2v_model = m.train_word2vec(w, embed_size, lr)
                m.save_embeddings(w2v_model, path_to_save)
                corr = t.test_embeddings(test_path, path_to_save)
                # Add Spearman correlation of current configuration to dictionary
                results.update({path_to_save: (corr)})

    print(results)
    # Print configuration with maximum Spearman Correlation
    print("Embeddings with best Spearman is ", max(results.items(), key=operator.itemgetter(1))[0])
    return

if __name__ == '__main__':

    # Parse the command line arguments
    args = parse_args()

    # Instantiate an instance of the 'Procesor', 'Model', and 'Tester' class
    p = Processor()
    m = Model()
    t = Tester()

    # If the processed training sentences exist, train the model and save it
    if os.path.exists(p.processed_pth):
        print("Found Pre-pocessed Dataset! Runing Grid Search Algorithm..")
        grid_search(p, m, t, args.test_f_path)
    # Else, instruct user to run the "preprocess" code first
    else:
        print("No Processed Dataset! Please run 'preprocess_ES.py [parent_path]'")
        print("where:")
        print("parent_path: Path to the eurosense.v1.0.high-precision.xml file")
