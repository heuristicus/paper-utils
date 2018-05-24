#!/usr/bin/env python

import sys
import os
import re
import operator
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class CitationType(Enum):
    SQUARE=1
    DOTTED=2
    PLAIN=3

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
    # possible conflicts with years in the citation text. Sometimes the citation
    # numbers are separated from the text, in which case they will be alone on a
    # line with the end of the line directly after the period. If not, a space
    # will separate the number and the citation text.
    dotted_citation_re = re.compile("^([0-9]{1,3})\.( |$)")

    def __init__(self, fname):
        self.name = os.path.basename(fname)
        self.fname = fname
        self.references_start = 0
        self.references_end = 0
        self.max_citation_num = 0
        self.matching = []
        # Contains all found citations, with 2-tuple containing citation number and line number
        self.square_citations = []
        self.dotted_citations = []
        self.all_citations = []

        with open(fname) as f:
            for lineno, line in enumerate(f):
                # sometimes the r and eferences get separated by a space (like, in a lot of cases)
                if "references" in line.lower():
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
                    self.square_citations.append((int(match.group(1)), lineno))
                    # Also check the number of the citation so we can know how
                    # many citations there are in the paper
                    if int(match.group(1)) > int(self.max_citation_num):
                        self.max_citation_num = int(match.group(1))

                dotted = self.dotted_citation_re.match(line)
                if dotted:
                    self.dotted_citations.append((int(dotted.group(1)), lineno))

        # Dotted citations are relatively rare, most papers use square brackets.
        # If they are used, then they only appear in the references section
        # Assume that any number of dotted citations below half the total number
        # of citations are just noise. Some papers with low max citation counts
        # might have a weird effect, so assume that if there aren't more than 10
        # dotted citations that this is a paper with square citations
        if len(self.dotted_citations) < max(self.max_citation_num/2, 10):
            self.all_citations = self.square_citations
            self.citation_type = CitationType.SQUARE
        else:
            self.all_citations = sorted(self.dotted_citations + self.square_citations, key=operator.itemgetter(1))
            self.citation_type = CitationType.DOTTED

        self.compute_sequences()
        self._get_references_end()
        self._get_references_text()

    def __str__(self):
        str_rep = ""
        str_rep += self.name + "\n"
        str_rep += "References start at {}, end at {}\n".format(self.references_start, self.references_end)
        str_rep += "Total citations: {}\n".format(self.max_citation_num)
        str_rep += "Square refs: {}, dotted refs: {}\n".format(len(self.square_citations), len(self.dotted_citations))

        return str_rep

    def compute_sequences(self, min_length=3, max_gap=8):
        """min_length is the minimum sequence length that will be considered

        max_gap is the maximum gap between two citations. We want to use this
        function to extract references from the references section rather than
        the body of the paper, and those tend to be pretty close to each other.

        """
        self.increasing_sequences = []
        if len(self.square_citations) < 2:
            return

        # 3-tuple, sequence length, start line, end line
        start_ind = 0 # this starting index means that the first sequence is one shorter than it should be?
        start_line = self.all_citations[0][1]

        for ind, item in enumerate(self.all_citations[1:]):
            # indexing into the original list with ind will give us the previous
            # item, since we are slicing the enumerated list from 1.
            prev_item = self.all_citations[ind]
            # Strictly less, because we really want to find sequences from the
            # references section which should be strictly increasing.
            if prev_item[0] < item[0] and item[1] - prev_item[1] < max_gap:
                if not start_ind:
                    start_ind = ind-1
                    start_line = prev_item[1]
            else:
                if start_ind and ind+1 - start_ind > min_length:
                    # citations per line, might be useful
                    num_lines = ind - start_ind
                    if item[1] == start_line: # the sequence ends on a single line
                        cpl = num_lines
                    else:
                        cpl = num_lines/float(item[1]-start_line)
                    self.increasing_sequences.append((num_lines, start_line, prev_item[1], cpl))

                start_ind = None # reset start ind so we know the sequence ended

        # only if the last sequence starts with an element in the list before
        # the last one. start_ind is none if there is no sequence to be
        # finalised
        if not start_line == self.all_citations[-1][1] and start_ind is not None:
            # -1 because len gives number of objects which starts from 1, we want inds
            num_lines = len(self.all_citations) - 1 - start_ind
            # Add the final sequence
            cpl = num_lines/float(self.all_citations[-1][1]-start_line)
            self.increasing_sequences.append((num_lines, start_line, self.all_citations[-1][1], cpl))

    def _get_references_end(self):
        """Estimate the point at which references for this paper end, based on the sequences extracted
        """
        after_refstart = self.sequences_after_line(self.references_start)
        if len(after_refstart) >= 1:
            self.references_end = sorted(after_refstart, key=operator.itemgetter(2), reverse=True)[0][2]
        else:
            self.references_end = None

    def _get_references_text(self):
        if not self.references_start:
            print("Didn't find start point of references, cannot get reference text")
            return

        lines = ""
        with open(self.fname) as f:
            # first need to get to the start of the references
            lines_processed = 0
            for line in f:
                lines_processed += 1
                if lines_processed >= self.references_start:
                    break

            lines_processed = 0
            # add a bit of padding so that we get the text of the last reference
            lines_to_process = self.references_end - self.references_start + 5 if self.references_end else None
            for line in f:
                lines += line
                lines_processed += 1
                if lines_to_process and lines_processed >= lines_to_process:
                    break

        return lines

    def sequences_after_line(self, lineno):
        after = []
        for seq in sorted(self.increasing_sequences, key=operator.itemgetter(1)):
            if lineno <= seq[1] or lineno <= seq[2]:
                after.append(seq)

        return after

def main():

    document_citations = []

    for fname in sys.argv[1:]:
        #print("Processing {}".format(os.path.basename(fname)))

        citation_group = CitationGroup(fname)

        document_citations.append(citation_group)

    for d in document_citations:
        if not d.increasing_sequences or not d.all_citations:
            print("no sequence/citations found for {}".format(d.name))
    print("papers with dotted citations:")
    for d in document_citations:
        if d.citation_type == CitationType.DOTTED:
            print(d)
            print(d._get_references_text())
            print("--------------------------------------------------")

if __name__ == '__main__':
    main()
