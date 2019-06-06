import os
import re

from argparse import ArgumentParser

from gensim.models import KeyedVectors

from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt

from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

# Specify command line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["test", "Test", "vis", "Vis", "plot", "Plot"], required=True, default="Test", help="Test the model or Visualize / Plot Embeddings")
    parser.add_argument("-t", "--test_f_path", required=False, default="", help="Path to the Testing (Secret) '.tab' file")
    parser.add_argument("-e", "--embed_path", required=False, default="../resources/embeddings.vec", help="Path to the sense embeddings")

    return parser.parse_args()

class Tester:
    def __init__(self):
        """
        Tester class that Tests the model on word similarity or Plots some embeddings
        """

        # Path to the '.vec' embeddings file to test
        # Path to a txt file where the results on Word Similarity will be written
        self.sim_pth = os.path.join("../resources", "similarity_score.txt")

    def load_embeddings(self, path_to_load):
        """
        Loads the '.vec' file from path specified as argument on the command line

        returns: loaded Word2Vec model
        """
        print()
        print("Loading embeddings from ", path_to_load)
        model = KeyedVectors.load_word2vec_format(path_to_load, binary=False)
        print("Successfully loaded embeddings!")
        return model


    def test_embeddings(self, parent, path_to_load):
        """
        Tests the embedding (from the '.vec' file) on a Word Similarity task

        returns: Value of the Spearman Correlation
        """
        # Ensure we can load the embeddings from the specified path
        if os.path.exists(path_to_load):
            print("\nFound the embeddings!")
            model = self.load_embeddings(path_to_load)
        else:
            print("\nUnable to find embeddings! Please verify path is correct or train the network to obtain them...")
            return

        # Obtain list of model's Vocabulary
        embeddings_vocab = list((model.wv.vocab))

        # Initialize list to store word pairs and human chosen values
        ground_truth = []
        word_pairs = []

        with open(parent, "r") as test_file:
            content = test_file.readlines()
            # Remove the first line as it is just a title for the file
            content.pop(0)
            for idx, line in enumerate(content):
                # Remove the '\n'
                line = line.rstrip()
                # Append human-chosen value to ground_truth list
                ground_truth.append(float(line.split("\t")[-1]))
                # Append word pairs to word_pairs list
                word_pairs.append(line.split("\t")[:2])

        # Create a 'predictions' list that has same size as the ground_truth
        predictions = [-1] * len(ground_truth)

        for p_idx, pair in enumerate(word_pairs):
            # For each word in the word pair
            # Find all senses that have the exact word in them
            all_senses_1 = [s for s in embeddings_vocab if s.split("_bn:")[0]==pair[0].lower()]
            all_senses_2 = [s for s in embeddings_vocab if s.split("_bn:")[0]==pair[1].lower()]

            # Sanity check, Ensure predictions for current pair are initialized to -1
            predictions[p_idx] = -1

            # Loop over all senses of each word and compute max cosine similarity
            if all_senses_1 != [] and all_senses_2 != []:
                for sense1 in all_senses_1:
                    for sense2 in all_senses_2:
                        predictions[p_idx] = max(predictions[p_idx], cosine_similarity(model[sense1].reshape(1, -1), model[sense2].reshape(1, -1))[0][0])

                # Print the word pair and the max cosine similarity calculated for this pair
                print(pair[0], " - ", pair[1], " - " , predictions[p_idx])

                # Write results of Word Similarity test to 'similarity_score.txt'
                with open (self.sim_pth, "a", encoding="utf8") as f:
                    f.write(pair[0] + " - " + pair[1] + " - " + str(predictions[p_idx]))
                    f.write("\n")
                f.close()

        # Sanity check, ensure sizes of predictions & ground_truth are the same
        # before passing them to the Spearman Correlation function
        assert len(predictions) == len(ground_truth)

        # Calculate the Spearman Correlation & p-value
        corr, p = spearmanr(ground_truth, predictions)

        # Print the results
        print("\nSpearman correlation is", corr)
        print("p-value is ", p)

        # Write the Spearman Correlation to 'similarity_score.txt'
        with open (self.sim_pth, "a", encoding="utf8") as f:
            f.write("Spearman correlation is " + str(corr))
            f.write("\n")
        f.close()

        return corr

    def visualize_embeddings(self, path_to_load):
        """
        Plots the embeddings (from the ".vec" file) for a few select 'representative' senses
        """

        # Ensure we can load the embeddings from the specified path
        if os.path.exists(path_to_load):
            print("\nFound the embeddings!")
            w2v_model = self.load_embeddings(path_to_load)
        else:
            print("\nUnable to find embeddings! Please verify path is correct or train the network to obtain them...")
            return

        # Retrive the model's Vocabulary
        words = list(w2v_model.wv.vocab)
        # Create list of words to be plotted
        list_of_words = ["King", "President", "Rome", "Italy", "Paris", "France", "Europe", "united_states_of_america", "country", "city"]
        # Find all senses of the words to be plotted
        gen = []
        for word_to_plot in list_of_words:
            gen.append([s for s in words if s.split("_bn:")[0]==word_to_plot.lower()])
        all_senses_to_plot = [item for sublist in gen for item in sublist]
        # Print the senses found
        print("Plotting the embeddings for the following senses {:}".format(all_senses_to_plot))

        # Get vectors all the senses we are interested in plotting
        X = w2v_model[all_senses_to_plot]
        
        # Preform PCA dimensionality reduction on the sense embedding vectors
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)

        # Create scatter plot of sense embeddings
        plt.scatter(result[:, 0], result[:, 1])

        # Loop over each vector / scatter point and annotate it with the sense
        for i, word in enumerate(all_senses_to_plot):
        	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

        # Plot the figure
        plt.title("Embeddings")
        plt.show()
        # Save the figure
        plt.savefig('my_embeddings.png', dpi=None, facecolor='w', edgecolor='b',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches="tight", pad_inches=0.1,
        frameon=None, metadata=None)


if __name__ == '__main__':

    # Parse the command line arguments
    args = parse_args()

    # Instantiate an instance of the 'Tester' class
    t = Tester()

    # Check if the model has already been tested before & results written to file
    if os.path.exists(t.sim_pth) and args.mode.lower() == "test":
        print("\nModel has already been tested!")
        print("Check the {:} file for detailed results".format(t.sim_pth))
        print("Will now display the Spearman Correlation present in the {:} file...".format(t.sim_pth))
        with open(t.sim_pth) as last_test_path:
            ltp_content = last_test_path.readlines()[-1]
            print(ltp_content)
    # If model hasn't been tested before, test it or plot it
    else:
        if args.mode.lower() == "test":
            print("\nTesting the model on Word Similarity Task...")
            t.test_embeddings(args.test_f_path, args.embed_path)
        elif args.mode.lower() == "vis" or args.mode.lower() == "plot":
            t.visualize_embeddings(args.embed_path)
