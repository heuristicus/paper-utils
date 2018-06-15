#!/usr/bin/env python

import random
import unittest
from reference_graph import arrays_contain_same_reference

paper_title_similar_pairs = [(['semantic', 'image', 'segmentation', 'with', 'deep', 'convolutional', 'nets', 'fully', 'connected', 'crfs'],
                              ['deeplab', 'semantic', 'image', 'segmentation', 'with', 'deep', 'convolutional', 'nets', 'atrous', 'convolution', 'fully', 'connected', 'crfs']),
                             (['pascal', 'visual', 'object', 'classes', 'voc', 'challenge'],
                              ['pascal', 'visual', 'object', 'classes', 'challenge', '2012', 'voc2012', 'results'])]
paper_titles = [['whats', 'point', 'semantic', 'segmentation', 'with', 'point', 'supervision'],
                ['convolutional', 'random', 'walk', 'networks', 'semantic', 'image', 'segmentation'],
                ['fully', 'convolutional', 'networks', 'semantic', 'segmentation'],
                ['deeplab', 'semantic', 'image', 'segmentation', 'with', 'deep', 'convolutional', 'nets', 'atrous', 'convolution', 'fully', 'connected', 'crfs'],
                ['boxsup', 'exploiting', 'bounding', 'boxes', 'supervise', 'convolutional', 'networks', 'semantic', 'segmentation']]

junk = [['geoffrey', 'hinton', '2012'],
        ['trevor', 'darrell'],
        ['anton', 'van', 'den', 'hengel', 'ian', 'reid'],
        ['torch7', 'matlab-like', 'environment', 'machine', 'learning', 'biglearn', 'nips', 'workshop', 'dai', 'sun', '2015'],
        ['volume', 'jmlr', 'proceedings', 'pages', '195-206', 'jmlrorg', '2012', 'karen', 'simonyan', 'andrew', 'zisserman'],
        ['url', 'http', '//wwwpascal-networkorg/challenges/', 'voc/voc2012/workshop/index'],
        ['pereira', 'burges', 'cjc', 'bottou', 'weinberger', 'eds'],
        ['pratikakis', 'dupont', 'ovsjanikov', 'editors', 'eurographics', 'workshop', 'object', 'retrieval', 'eurographics', 'association'],
        ['lafferty', 'williams', 'cki', 'shawe-taylor', 'zemel', 'culotta', 'eds'],
        ['navab', 'hornegger', 'wells', 'frangi', 'eds', 'medical', 'image', 'computing', 'computer-assisted', 'intervention', 'miccai', '2015'],
        ['3rd', 'workshop', 'semantic', 'perception', 'mapping', 'exploration', 'spme'],
        ['isprs', 'annals', 'photogrammetry', 'remote', 'sensing', 'spatial', 'information', 'sciences']]

class TestArraySimilarityMethods(unittest.TestCase):


    
    def test_noclutter(self):
        for ind, title in enumerate(paper_titles):
            common = set(title).intersection(set(title))
            self.assertTrue(arrays_contain_same_reference(title, title, common))

    def test_noclutter_invalid(self):
        first = paper_titles[0]
        second = paper_titles[1]
        common = set(first).intersection(set(second))

        self.assertFalse(arrays_contain_same_reference(first, second, common))

    def test_noclutter_similar(self):
        similar_ind = random.randint(0, len(paper_title_similar_pairs) - 1)
        
        first = paper_title_similar_pairs[similar_ind][0]
        second = paper_title_similar_pairs[similar_ind][1]
        common = set(first).intersection(set(second))

        self.assertFalse(arrays_contain_same_reference(first, second, common))

    def test_noclutter_cutoff(self):
        # What do we want to be the case here? If we remove the last word in one
        # of the titles then it's sort of like the other one has junk at the
        # end?
        first = paper_titles[0]
        second = paper_titles[1][:-1]
        common = set(first).intersection(set(second))

        # Go with a more inclusive assumption for now...
        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_before(self):
        paper_ind = random.randint(0, len(paper_titles) - 1)
        junk_ind = random.randint(0, len(junk) - 1)
        
        first = paper_titles[paper_ind]
        second = junk[junk_ind] + paper_titles[paper_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

        # check it works if they are the other way round
        first = junk[junk_ind] + paper_titles[paper_ind]
        second = paper_titles[paper_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_before_invalid(self):
        paper_ind = random.randint(0, len(paper_titles) - 1)
        paper_ind_different = paper_ind - 1 if paper_ind > 0 else paper_ind + 1
        junk_ind = random.randint(0, len(junk) - 1)
        
        first = paper_titles[paper_ind]
        second = junk[junk_ind] + paper_titles[paper_ind_different]
        common = set(first).intersection(set(second))

        self.assertFalse(arrays_contain_same_reference(first, second, common))

        # check it works if they are the other way round
        first = junk[junk_ind] + paper_titles[paper_ind_different]
        second = paper_titles[paper_ind]
        common = set(first).intersection(set(second))

        self.assertFalse(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_before_similar(self):
        similar_ind = random.randint(0, len(paper_title_similar_pairs) - 1)
        junk_ind = random.randint(0, len(junk) - 1)
        
        first = paper_title_similar_pairs[similar_ind][0]
        second = junk[junk_ind] + paper_title_similar_pairs[similar_ind][1]
        common = set(first).intersection(set(second))

        self.assertFalse(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_after(self):
        paper_ind = random.randint(0, len(paper_titles) - 1)
        junk_ind = random.randint(0, len(junk) - 1)
        
        first = paper_titles[paper_ind]
        second = paper_titles[paper_ind] + junk[junk_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

        # First paper has junk rather than second
        first = paper_titles[paper_ind] + junk[junk_ind]
        second = paper_titles[paper_ind] 
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_after_similar(self):
        similar_ind = random.randint(0, len(paper_title_similar_pairs) - 1)
        junk_ind = random.randint(0, len(junk) - 1)
        
        first = paper_title_similar_pairs[similar_ind][0]
        second = paper_title_similar_pairs[similar_ind][1] + junk[junk_ind]
        common = set(first).intersection(set(second))

        self.assertFalse(arrays_contain_same_reference(first, second, common))
    
    def test_one_clutter_after_repeatword(self):
        # use clutter which contains a word which exists in the paper title
        paper = ['whats', 'point', 'semantic', 'segmentation', 'with', 'point', 'supervision']
        junk = ['3rd', 'workshop', 'semantic', 'perception', 'mapping', 'exploration', 'spme']
        
        first = paper
        second = paper + junk
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

        first = paper + junk
        second = paper
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_before_repeatword(self):
        paper = ['deeplab', 'semantic', 'image', 'segmentation', 'with', 'deep', 'convolutional', 'nets', 'atrous', 'convolution', 'fully', 'connected', 'crfs']
        junk = ['navab', 'hornegger', 'wells', 'frangi', 'eds', 'medical', 'image', 'computing', 'computer-assisted', 'intervention', 'miccai', '2015']

        first = paper
        second = junk + paper
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_before_after_repeatword_before(self):
        paper = ['deeplab', 'semantic', 'image', 'segmentation', 'with', 'deep', 'convolutional', 'nets', 'atrous', 'convolution', 'fully', 'connected', 'crfs']
        junk_before = ['navab', 'hornegger', 'wells', 'frangi', 'eds', 'medical', 'image', 'computing', 'computer-assisted', 'intervention', 'miccai', '2015']
        junk_after = ['volume', 'jmlr', 'proceedings', 'pages', '195-206', 'jmlrorg', '2012', 'karen', 'simonyan', 'andrew', 'zisserman']

        first = paper
        second = junk_before + paper + junk_after
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_one_clutter_before_after_repeatword_after(self):
        paper = ['deeplab', 'semantic', 'image', 'segmentation', 'with', 'deep', 'convolutional', 'nets', 'atrous', 'convolution', 'fully', 'connected', 'crfs']
        junk_before = ['torch7', 'matlab-like', 'environment', 'machine', 'learning', 'biglearn', 'nips', 'workshop', 'dai', 'sun', '2015']
        junk_after = ['3rd', 'workshop', 'semantic', 'perception', 'mapping', 'exploration', 'spme']

        first = paper
        second = junk_before + paper + junk_after
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

        
    def test_one_clutter_before_after(self):
        paper_ind = random.randint(0, len(paper_titles) - 1)
        junk_before_ind = random.randint(0, len(junk) - 1)
        junk_after_ind = random.randint(0, len(junk) - 1)
        
        while junk_before_ind == junk_after_ind:
            junk_after_ind = random.randint(0, len(junk) - 1)
        
        first = paper_titles[paper_ind]
        second = junk[junk_before_ind] + paper_titles[paper_ind] + junk[junk_after_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

        # First paper has junk rather than second
        first = junk[junk_before_ind] + paper_titles[paper_ind] + junk[junk_after_ind]
        second = paper_titles[paper_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_both_clutter_before(self):
        paper_ind = random.randint(0, len(paper_titles) - 1)
        junk_first_ind = random.randint(0, len(junk) - 1)
        junk_second_ind = random.randint(0, len(junk) - 1)
        
        while junk_first_ind == junk_second_ind:
            junk_second_ind = random.randint(0, len(junk) - 1)
        
        first = junk[junk_first_ind] + paper_titles[paper_ind]
        second = junk[junk_second_ind] + paper_titles[paper_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_both_clutter_after(self):
        paper_ind = random.randint(0, len(paper_titles) - 1)
        junk_first_ind = random.randint(0, len(junk) - 1)
        junk_second_ind = random.randint(0, len(junk) - 1)
        
        while junk_first_ind == junk_second_ind:
            junk_second_ind = random.randint(0, len(junk) - 1)
        
        first = paper_titles[paper_ind] + junk[junk_first_ind]
        second = paper_titles[paper_ind] + junk[junk_second_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))


    def test_both_clutter_after_repeated_different(self):
        # Artificially construct something so that the overlap is in the junk
        paper = ["this", "is", "the", "paper", "title"]
        junk_first = ["this", "here", "junk", "words"]
        junk_second = ["yet", "more", "stuff", "junk"]
        first = paper + junk_first
        second = paper + junk_second
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

    def test_both_clutter_before_after(self):
        paper_ind = random.randint(0, len(paper_titles) - 1)
        junk_before_first_ind = random.randint(0, len(junk) - 1)
        junk_before_second_ind = random.randint(0, len(junk) - 1)
        junk_after_first_ind = random.randint(0, len(junk) - 1)
        junk_after_second_ind = random.randint(0, len(junk) - 1)
        
        while junk_before_first_ind == junk_before_second_ind:
            junk_before_second_ind = random.randint(0, len(junk) - 1)

        while junk_after_first_ind == junk_after_second_ind:
            junk_after_second_ind = random.randint(0, len(junk) - 1)

        
        first = junk[junk_before_first_ind] + paper_titles[paper_ind] + junk[junk_after_first_ind]
        second = junk[junk_before_second_ind] + paper_titles[paper_ind] + junk[junk_after_second_ind]
        common = set(first).intersection(set(second))

        self.assertTrue(arrays_contain_same_reference(first, second, common))

if __name__ == '__main__':
    unittest.main()
