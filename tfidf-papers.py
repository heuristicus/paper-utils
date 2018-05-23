#!/usr/bin/env python

import math
import string
import sys
import re
import operator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import igraph

def naive_cosine_similarity(a, b):
    apsqrt = np.sqrt(np.sum(np.power(a, 2)))
    bpsqrt = np.sqrt(np.sum(np.power(b, 2)))

    return np.dot(a, b)/(apsqrt*bpsqrt)

class Document(object):
    def __init__(self, filename):
        # Key is word, value is count of word in this document
        self.words = {}
        self.name = os.path.basename(filename)
        
        with open(filename, 'r') as f:
            # split on spaces, assume this gives words (or most words)
            for word in f.read().replace('\n', ' ').split(' '):
                # allow hyphenated words
                stripped = re.sub('[^a-zA-Z-]', '', word).lower()
                # ignore long and short strings, and stuff which starts with
                # hyphen, since that is not useful stuff
                if 1 < len(stripped) < 20 and not stripped[0] == '-':
                    if stripped in self.words:
                        self.words[stripped] += 1
                    else:
                        self.words[stripped] = 1

    def get_full_word_dict(self, all_words):
        """In the document itself we only store information for the words contained in
        the document, but we also need to be able to get a dict with all the
        words in it for ease of creating a matrix later

        """
        full_dict = dict(self.words)
        for word in all_words:
            if word not in full_dict:
                full_dict[word] = 0

        return full_dict
                
    def get_tfidf_vector(self, all_words_idf_dict):
        """Get the full tfidf vector for this document which has a tfidf value for all
        words in the document group rather than just for the words in this
        document (most words will not be in the document, so there will be a lot of zeros)

        """
        vec = []
        # the vectors will have tfidf for words in the document group in
        # alphabetical order
        for word in sorted(all_words_idf_dict):
            if word in self.words:
                # this document holds term frequency, the dict we receive holds
                # inverse document frequency
                vec.append(self.words[word] * all_words_idf_dict[word])
            else:
                vec.append(0) # if term frequency is zero then so is tfidf

        return np.mat(vec)

    def get_short_name(self):
        """Assumes that the papers are named according to author - year - title
        """
        split = self.name.split(' - ')
        # author, year, and first couple of words of paper title
        return "{} ({}), {}".format(split[0], split[1], " ".join(split[2].split(' ')[:3]))

class DocumentGroup(object):
    def __init__(self, documents):
        self.documents = documents

        # key is word, value is tfidf
        self.idf = {}
        word_set = set()
        for doc in self.documents:
            word_set.update(list(doc.words.keys()))

        print("Total words in document group: {}".format(len(word_set)))

        print("Computing idf")
        for word in word_set:
            count = 0
            for doc in self.documents:
                if word in doc.words:
                    count += 1

            self.idf[word] = math.log(len(self.documents)/count)

        print("Creating tfidf matrix")
            
        # this matrix will have the tfidf vector for each document in columns.
        # Terms are rows. We know the size so preallocate.
        docgroup_mat = np.zeros((len(self.idf), len(self.documents)))
        for col, doc in enumerate(self.documents):
            docgroup_mat[:,col] = doc.get_tfidf_vector(self.idf)

        # transpose the docgroup matrix, cosine similarity from scikit assumes
        # samples are in rows
        docgroup_sparse = sparse.csr_matrix(np.transpose(docgroup_mat))

        self.similarity = cosine_similarity(docgroup_sparse)
        
        # # Computing from dense matrix
        # print("Computing similarity matrix")

        # similarity = np.zeros((len(self.documents), len(self.documents)))
        # for i in range(0, len(self.documents)):
        #     for j in range(i, len(self.documents)):
        #         similarity[i, j] = naive_cosine_similarity(docgroup_mat[:, i], docgroup_mat[:, j])
        #         similarity[j, i] = similarity[i, j]

        self._make_graph()

    def print_topn(self, n):
        for doc_ind, doc in enumerate(self.documents):
            print("Looking at top {} documents similar to doc #{}, {}".format(n, doc_ind, doc.name))
            for rank, col in enumerate(reversed(np.argsort(self.similarity[doc_ind,:])[-n-1:-1])):
                print("Rank {} ({:.2f}), #{}: {}".format(rank, self.similarity[doc_ind, col], col, self.documents[col].name))
            print("--------------------")

    def _make_graph(self):
        print("Creating graph")
        edges = []
        weights = []
        
        for col in range(0, self.similarity.shape[1]):
            # cutoff based on percentile of the column
            cutoff = np.percentile(self.similarity[:, col], 90)
            no_self_similarity = np.concatenate((self.similarity[:col, col], self.similarity[col+1:, col]))
            norm_no_self_similarity = no_self_similarity/max(no_self_similarity)
            
            # Only add edges once. We go over the values in the lower triangle
            # of the matrix
            for row in range(col+1, self.similarity.shape[0]):
                if self.similarity[row,col] > cutoff:
                    weights.append(norm_no_self_similarity[row-1])
                    edges.append((row, col))

        self.graph = igraph.Graph()
        self.graph.add_vertices(len(self.documents))
        self.graph.add_edges(edges)
        self.graph.vs["name"] = [doc.get_short_name() for doc in self.documents]
        self.graph.vs["label"] = range(0, len(self.documents))
        self.graph.es["weight"] = weights

    def show_graph(self):
        igraph.plot(self.graph, layout=self.graph.layout("fr", weights="weight"))
                
def main():
    args = sys.argv
    docs = []
    for doc_fname in args[1:]:
        docs.append(Document(doc_fname))

    docgroup = DocumentGroup(docs)

    docgroup.print_topn(5)
    
    docgroup.show_graph()

if __name__ == '__main__':
    main()
            
        
