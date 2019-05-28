import os
import numpy as np
import pickle
from argparse import ArgumentParser

from tqdm import tqdm
import xml.etree.ElementTree as ET


import multiprocessing
from time import time
from gensim.models import Word2Vec

from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as pyplot

from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=['Train','Dev','Test'], help="Train, Dev, or Test")
    parser.add_argument("parent_path", help="The path of the Training (Eurosense) or Testing (Secret) Data folder")

    return parser.parse_args()

class Pocessor:
    def __init__(self, m):

        self.mode = m

        if m == (""):
            pass
        else:
            print()
            print("Creating Necessary Files & Folders")
            folder = "./Processed_" + self.mode
            if not os.path.exists(folder):
                os.makedirs(folder)

            self.processed_pth = os.path.join(folder, (self.mode + "_processed.txt"))
            self.dict_pth = os.path.join(folder, (self.mode + "_annot_dictionary.pickle"))
            self.embed_pth = os.path.join("../resources", "embeddings.vec")

    def open_xml(self, parent):
        files = ["eurosense_500.xml"]
        print("{:} Dataset will be built from the following file(s) {:}...".format(self.mode, files))
        # print(parent)

        no_sp = open(self.processed_pth , "w", encoding="utf8")
        l = []
        d = {}

        for idx, file in enumerate(files):
            # context = ET.iterparse(parent, events=("start", "end"))
            # context = iter(context)
            # event, root = context.next()
            c = 0
            flag = True
            expecting_annotation = False

            for event, elem in ET.iterparse(parent, events=("start", "end")):
                # print(c)

                if event == 'start' :
                    # print("This current element is", elem)

                    if elem.tag == 'sentence' :
                         print(elem.get("id"))
                        # d = {}
                    if elem.tag == 'text' and elem.text != None:
                        # print(elem.get("lang"))
                        d = {}
                        if elem.get("lang") == "en":

                            if expecting_annotation == True:
                                l.append(None)
                            elif expecting_annotation == False:
                                expecting_annotation = True

                            flag = True
                            # print((elem.text))
                            # print(type(elem.text))
                            with open (self.processed_pth, "a", encoding="utf8") as f:
                                f.write(elem.text)
                                f.write("\n")
                                c +=1
                            f.close()
                            # print("Value of 'text' tag is %s" % elem.text.encode('utf-8'))
                            # print(elem.text.encode('utf-8'))
                            # print(elem.text)
                            # print(list(elem.text))

                    if elem.tag == 'annotation' and elem.get("lang") == "en":
                        if expecting_annotation == True:
                            expecting_annotation = False
                            
                        if elem.text == None:
                            if flag == True:
                                l.append(None)
                                flag = False
                        # print(elem.get("anchor"))
                        # print(elem.get("lemma"))
                        # print(elem.text)
                        else:
                            label = elem.get("lemma") + "_" + elem.text
                            # print(label)
                            # key = elem.get("anchor")
                            d.update({elem.get("anchor"): (elem.get("lemma") + "_" + elem.text)})
                            # print("Dictionary of 'root':'annotation for each sentence is'", d)
                            if flag == True:
                                l.append(d)
                                flag = False
                # if event == 'end':
                #     print()
                elem.clear()

            # print(d)
            # Save dictionary here
            with open(self.dict_pth, "wb") as handle:
                pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print()
            print()
            # print(l)
            print()
            # print(l[0])
            # print(l[1])
            # print(l[2])
            # print(l[3])
            # print(l[4])
            # print(l[5])
            print("Number of sentences I wrote in th file is ", c)
            print("Length of list of parsed sentences is ", len(l))


    def create_training(self):
        print()
        print("Creating Word2Vec Training Data...")

        assert os.path.exists(self.dict_pth), "Training Dictionary does not exist"

        with open(self.dict_pth, 'rb') as handle:
            annotations_l = pickle.load(handle)
            print("Expected lengths of text file", len(annotations_l))
            # print(annotations_l)
        print("Done Loading the list of dictionaries!")

        with open(self.processed_pth, 'r', encoding="utf8") as f:
            f_content = f.readlines()

        # for d in annotations_l:
        #     # print(l)
        #     for key, value in d.items():
        #         # print(value)
        train = []
        for idx, sentence in enumerate(f_content):
            # print("Sentence {} number {}".format(idx, sentence))
            if annotations_l[idx] != None:
                for key, value in annotations_l[idx].items():
                # print(sentence)
                # if key in sentence:
                    # print("Found!", key)
                    # print(sentence.find(key))
                    # print(key)
                    # print(value)
                    sentence = sentence.replace(key, value)
            # print(sentence)

            sentence = sentence.rstrip()
            train.append([])
            for word in sentence.split():
                train[idx].append(word)

        # print(train)
        print("Lenth of Training Data is", len(train))

        return train

    def train_word2vec(self):
        training_data = self.create_training()
        cores = multiprocessing.cpu_count() # Count the number of cores in a computer

        print()
        print("Training Word2Vec Model....")
        print("Number of cores is ", cores)
        w2v_model = Word2Vec(min_count=1, window=2, size=300, sample=6e-5, #change min_count !
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

        t = time()

        w2v_model.build_vocab(training_data, progress_per=10000)
        print(w2v_model)
        # words = list(w2v_model.wv.vocab)
        # print(words)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()

        w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

        return w2v_model

    def visualize_embeddings(self):
        w2v_model = self.train_word2vec()
        X = w2v_model[w2v_model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        pyplot.scatter(result[:, 0], result[:, 1])
        words = list(w2v_model.wv.vocab)
        for i, word in enumerate(words):
        	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()

    def save_embeddings(self):
        w2v_model = self.train_word2vec()
        X = w2v_model[w2v_model.wv.vocab]
        # print(X)
        print()
        print("Saving Embeddings Matrix....")
        print("Embeddings Matrix type is ", type(X))
        print("Embeddings Matrix size ", X.shape)
        # words = list(w2v_model.wv.vocab)
        words = w2v_model.wv.vocab
        # print(type(words))
        print("Model's Vocab size ", len(words))
        # print(words)
        with open(self.embed_pth, "w") as f:
            f.write(str(X.shape[0]) + " " + str(X.shape[1]))
            f.write("\n")
        f.close()
        for key, value in words.items():
            with open(self.embed_pth, "a") as f:
                f.write(key + " ")
                for element in w2v_model[key]:
                    f.write(str(element) + " ")
                f.write("\n")
            f.close()
            # print("The key is ", key)
            # print("The vale is ", value)
            # print("The vale is ", type(value))
            # print("The value is ", w2v_model[key])
        print()
        print("Successfully Saved embeddings!")
        return

    def load_embeddings(self):
        model = KeyedVectors.load_word2vec_format(self.embed_pth, binary=False)
        print()
        print("Successfully loaded embeddings!")
        # print(model)
        # print(model[model.wv.vocab])
        # print(model[model.wv.vocab].shape)
        # print((model.wv.vocab).keys())
        return model

    def test_embeddings(self, parent):
        model = self.load_embeddings()
        # print(model)
        # print(model[model.wv.vocab])
        # print(model[model.wv.vocab].shape)
        embeddings_vocab = list((model.wv.vocab).keys())
        # print(list((model.wv.vocab).keys()))
        # print(model["the"])

        ground_truth = []
        word_pairs = []

        with open(parent, "r") as test_file:
            content = test_file.readlines()
            # print(content)
            # print(type(content))
            content.pop(0)
            for idx, line in enumerate(content):
                # print("Single line in the test file", line)
                line = line.rstrip()
                ground_truth.append(float(line.split("\t")[-1]))
                word_pairs.append(line.split("\t")[:2])


        predictions = [None] * len(ground_truth)
        # print("Length of test word pairs", len(word_pairs))
        # print("Length of predictions", len(predictions))
        #
        # print("word pairs", word_pairs)
        # print("ground truth", ground_truth)

        # """ This assumes that each word has ONLY ONE Sense / Sense embedding"""
        # for p_idx, pair in enumerate(word_pairs):
        #     print(pair)
        #     print(pair[0], pair[1])
        #     try:
        #         # print(model[pair[0]], model[pair[1]])
        #         predictions[p_idx] = (cosine_similarity(model[pair[0]], model[pair[1]]))
        #     except:
        #         print("Word not found in my embeddings")
        #         predictions[p_idx] = (-1)
        #         continue

        """ This assumes that each word has MANY Senses / Sense embedding"""
        for p_idx, pair in enumerate(word_pairs):
            # print(pair)
            # print(pair[0], pair[1])
            """If the word is in my embeddings vocab"""
            # print("Word 0 found", pair[0] in embeddings_vocab)
            # print("Word 1 found", pair[1] in embeddings_vocab)

            """Find all senses of the word """
            all_senses_1 = [s for s in embeddings_vocab if pair[0]+"_bn" in s]
            all_senses_2 = [s for s in embeddings_vocab if pair[1]+"_bn" in s]

            """The approach below returned 'institiutinal' as a sense for the word 'institution' """
            # all_senses_1 = [s for s in embeddings_vocab if pair[0] in s]
            # all_senses_2 = [s for s in embeddings_vocab if pair[1] in s]
            # print("All senses of word 1", all_senses_1)
            # print("All senses of word 2", all_senses_2)
            """Loop over all senses of word_1 and word_2 and compute cosine_similarity """
            predictions[p_idx] = -1
            if all_senses_1 != []:
                print("Senses for word 1 found at line ", p_idx+2, all_senses_1)
            if all_senses_2 != []:
                print("Senses for word 2 found at line ", p_idx+2, all_senses_2)
            if all_senses_1 != [] and all_senses_2 != []:
                print("Senses for both words found at line ", p_idx, all_senses_1, all_senses_2)
                for sense1, sense2 in zip(all_senses_1, all_senses_2):
                    predictions[p_idx] = max(predictions[p_idx], cosine_similarity(model[sense1], model[sense2])[0][0])

        # print(ground_truth)
        print(predictions)
        print(len(ground_truth))
        print(len(predictions))
        assert len(predictions) == len(ground_truth)
        # print(word_pairs)
        # print(len(ground_truth))
        predictions = ground_truth
        corr, p = spearmanr(predictions, ground_truth)
        print(corr)
        print(p)
        return



if __name__ == '__main__':

    args = parse_args()
    # mode = args.mode
    # parent = args.parent_path

    p = Pocessor(args.mode)

    if args.mode == ("Train") or args.mode == ("Dev"):

        p.open_xml(args.parent_path)
        # p.create_training()
        # p.train_word2vec()
        # p.visualize_embeddings()
        p.save_embeddings()
        p.load_embeddings()

    elif args.mode == ("Test"):
        p.test_embeddings(args.parent_path)

    # print("Parent is", args.parent_path)
