#!/usr/bin/env python

import sys
import os
import re
import operator
import matplotlib.pyplot as plt
import numpy as np

class CitationGroup(object):
    # regex for citation-like things starting at the beginning of a line, which
    # have length greater than 5 after the citation block (i.e. after [1]),
    # which may indicate a citation. This doesn't work if the citation block and
    # the text of the citation have been separated by the pdf->text conversion
    linestart_re = re.compile("^\[*[0-9]+(\]|\.).{5,}")
    # Gets all citations which are enclosed within square brackets
    citation_re = re.compile("\[([0-9]+)\]")
    # Some papers have citations in the form "1. " followed by the citation
    # text. The citation number is assumed to be below 1000, which filters out
    # possible conflicts with years in the citation text
    dotted_citation_re = re.compile("^([0-9]{1,3})\. ")
    
    def __init__(self, filehandle):
        self.references_start = 0
        self.max_citation_num = 0
        self.matching = []
        # Contains all found citations, with 2-tuple containing citation number and line number
        self.all_citations = []

        for lineno, line in enumerate(filehandle):
            if "references" in line.lower() or "citations" in line.lower():
                self.references_start = lineno

            # ignore short lines, usually just numbers from tables. But
            # sometimes the text conversion puts citation brackets in the
            # reference section onto their own lines separate from the
            # actual citation text
            if len(line) > 5 and self.linestart_re.match(line):
                self.matching.append((lineno, line))

            # Look for all square bracket citations in a line, and add the
            # line number to the list for each one found
            for match in self.citation_re.finditer(line):
                self.all_citations.append((int(match.group(1)), lineno))
                # Also check the number of the citation so we can know how
                # many citations there are in the paper
                if int(match.group(1)) > int(self.max_citation_num):
                    self.max_citation_num = int(match.group(1))

            dotted = self.dotted_citation_re.match(line)
            if dotted:
                self.all_citations.append((int(dotted.group(1)), lineno))

        self.compute_sequences()

    def compute_sequences(self, min_length=3):
        self.increasing_sequences = []
        print(self.all_citations)
        if len(self.all_citations) < 2:
            return

        # 3-tuple, sequence length, start line, end line
        start_ind = 0 # this starting index means that the first sequence is one shorter than it should be?
        start_line = self.all_citations[0][1]

        for ind, item in enumerate(self.all_citations[1:]):
            # indexing into the original list with ind will give us the previous
            # item, since we are slicing the enumerated list from 1.
            if self.all_citations[ind][0] <= item[0]:
                print("starting sequence at {}".format(item))
                if not start_ind:
                    start_ind = ind
                    start_line = item[1]
            else:
                if start_ind and ind+1 - start_ind > min_length:
                    add_tuple = (ind+1-start_ind, start_line, item[1])
                    # citations per line, might be useful
                    if item[1] == start_line: # the sequence ends on a single line
                        cpl = ind-start_ind
                    else:
                        cpl = (ind-start_ind)/float(item[1]-start_line)
                    self.increasing_sequences.append((ind-start_ind, start_line, item[1], cpl))

                start_ind = None # reset start ind so we know the sequence ended

        # only if the last sequence starts with an element in the list before
        # the last one. start_ind is none if there is no sequence to be
        # finalised
        if not start_line == self.all_citations[-1][1] and start_ind is not None:
            # Add the final sequence
            cpl = (len(self.all_citations)-start_ind)/float(self.all_citations[-1][1]-start_line)
            self.increasing_sequences.append((len(self.all_citations)-start_ind, start_line, self.all_citations[-1][1], cpl))

def main():

    document_citations = []
    
    for fname in sys.argv[1:]:
        print("Processing {}".format(os.path.basename(fname)))
        with open(fname) as f:
            citation_group = CitationGroup(f)

        if citation_group.references_start:
            print("Got possible references marker at line {}".format(citation_group.references_start))

        print(citation_group.all_citations)
        print("Citations sequence:\n{}".format([x[0] for x in citation_group.all_citations]))

        print("Increasing sequences in the citations (sequence length, start line, end line, citations per line):\n {}".format(sorted(citation_group.increasing_sequences, key=operator.itemgetter(3), reverse=True)))

        print("Apparent total citations: {}".format(citation_group.max_citation_num))

        print("--------------------")

        document_citations.append(citation_group)
                
if __name__ == '__main__':
    main()
