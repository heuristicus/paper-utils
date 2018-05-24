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
    TRIGRAPH=4

class CitationGroup(object):
    # Gets all citations which are enclosed within square brackets
    citation_re = re.compile("\[([0-9]+)\]")
    # Some papers have citations in the form "1. " followed by the citation
    # text. The citation number is assumed to be below 1000, which filters out
    # possible conflicts with years in the citation text. Sometimes the citation
    # numbers are separated from the text, in which case they will be alone on a
    # line with the end of the line directly after the period. If not, a space
    # will separate the number and the citation text.
    dotted_citation_re = re.compile("^([0-9]{1,3})\.(\s*|$)")

    # Some citations in the text itself will be in the form of (surname et al.
    # 2001b; surname and other 1999), this captures those (these are quite rare
    # though)
    named_re = re.compile("\(((?:[a-zA-Z0-9 \n\.,]+(?:[0-9]{4}[a-z]*)+[; ]*)+)\)")

    # The AMS authorship trigraph takes the form [FS+90, AB+91]. Assume that
    # they are usually comma separated if multiple citations in same bracket
    # pair
    trigraph_re = re.compile("\[((?:[A-Z+]+[0-9]{2}[, ]*)+)\]")

    # Some papers have end material which we should make sure to ignore when
    # looking at the references, these strings often appear at the start of
    # those sections.
    end_strings = ["appendix", "appendices", "supplementary"]

    def __init__(self, fname):
        self.name = os.path.basename(fname)
        self.fname = fname
        self.references_start = 0
        self.references_end = 0
        self.end_material_lines = []
        self.max_citation_num = 0
        # Contains all found citations, with 2-tuple containing citation number and line number
        self.square_citations = []
        self.dotted_citations = []
        self.named_citations = []
        self.trigraph_citations = []
        self.all_citations = []

        with open(fname) as f:
            for lineno, line in enumerate(f):
                lineno += 1 # enumeration starts at zero
                # sometimes the r and eferences get separated by a space (like, in a lot of cases)
                if "references" in line.lower() and len(line) < len("references") + 5:
                    self.references_start = lineno

                # try to get some information about possible supplementary
                # material after the references
                for end_string in self.end_strings:
                    # The supplementary sections usually have a title which is
                    # on its own line, so we might be able to ignore mentions in
                    # references or other parts of the paper where lines will be
                    # longer. This is usually only a problem with papers which
                    # use a plain citation scheme
                    if end_string in line.lower() and len(line) < len(end_string) + 10:
                        self.end_material_lines.append(lineno)

                # Look for all square bracket citations in a line, and add the
                # line number to the list for each one found
                for match in self.citation_re.finditer(line):
                    citation_number = int(match.group(1))
                    self.square_citations.append((citation_number, lineno))
                    # Also check the number of the citation so we can know how
                    # many citations there are in the paper
                    if citation_number > self.max_citation_num:
                        self.max_citation_num = citation_number

                for match in self.trigraph_re.finditer(line):
                    self.trigraph_citations.append((match.group(1), lineno))

                dotted = self.dotted_citation_re.match(line)
                if dotted:
                    citation_number = int(dotted.group(1))
                    self.dotted_citations.append((int(dotted.group(1)), lineno))
                    if citation_number > self.max_citation_num:
                        self.max_citation_num = citation_number

            # Named matches span lines, so need to do a search over the full document rather than lines
            if not self.dotted_citations or not self.square_citations or not self.trigraph_citations:
                f.seek(0)
                for match in self.named_re.finditer(f.read()):
                    self.named_citations.append((match.group(1), lineno))

        # If there are no citations to be found after the start of the
        # references, then it's likely that the citations are in plain style
        all_tmp = sorted(self.dotted_citations + self.square_citations, key=operator.itemgetter(1))
        if self.named_citations:
            # We do not compute a sequence for named citations as the regex does
            # not match the references section
            for item in self.named_citations:
                print(item[0].replace("\n", '').split('; '))

            self.all_citations = all_tmp
            self.citation_type = CitationType.PLAIN

        # Dotted citations are relatively rare, most papers use square brackets.
        # If they are used, then they only appear in the references section
        # Assume that any number of dotted citations below half the total number
        # of citations are just noise. Some papers with low max citation counts
        # might have a weird effect, so assume that if there aren't more than 10
        # dotted citations that this is a paper with square citations
        elif len(self.dotted_citations) < max(self.max_citation_num/2, 10):
            self.all_citations = self.square_citations
            self.citation_type = CitationType.SQUARE
            self.compute_sequences()
        elif self.trigraph_citations:
            # Trigraph and named citations do not contain information about the
            # number of citations which exist, so for consistency we must convert
            # the values we saw
            # Trigraph citations are not grouped
            tri_set = set()
            for item in self.trigraph_citations:
                tri_set.update(item[0].split(', '))

            self.max_citation_number = len(tri_set)
            self.all_citations = self.trigraph_citations
            self.citation_type = CitationType.TRIGRAPH
            self.compute_sequences(ignore_number=True)
        else:
            self.all_citations = all_tmp
            self.citation_type = CitationType.DOTTED
            self.compute_sequences()

        self._get_references_end()
        self._get_references_text()

    def __str__(self):
        str_rep = ""
        str_rep += self.name + "\n"
        str_rep += "References start at {}, end at {}\n".format(self.references_start, self.references_end)
        str_rep += "Total citations: {}\n".format(self.max_citation_num)
        str_rep += "Citation type seems to be {}\n".format(self.citation_type)
        str_rep += "Increasing sequences:\n {}\n".format(self.increasing_sequences)
        str_rep += "Square refs: {}, dotted refs: {}, named refs: {}, trigraph refs: {}\n".format(len(self.square_citations), len(self.dotted_citations), len(self.named_citations), len(self.trigraph_citations))

        return str_rep

    def compute_sequences(self, min_length=3, max_gap=8, ignore_number=False):
        """min_length is the minimum sequence length that will be considered

        max_gap is the maximum gap between two citations. We want to use this
        function to extract references from the references section rather than
        the body of the paper, and those tend to be pretty close to each other.

        ignore_number will just use the gap and min length rather than looking
        at the number of the citation to check if it is increasing. This is
        useful for the trigraph type which isn't numbered

        """
        self.increasing_sequences = []
        if len(self.all_citations) < 2:
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
            if (ignore_number or prev_item[0] < item[0]) and item[1] - prev_item[1] < max_gap:
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
                        cpl = num_lines/float(prev_item[1]-start_line)
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
            if self.references_end:
                lines_to_process = self.references_end - self.references_start + 5
            elif self.end_material_lines:
                lines_to_process = max(self.end_material_lines) - self.references_start
            else:
                lines_to_process = None

            for line in f:
                lines += line
                lines_processed += 1
                if lines_to_process and lines_processed >= lines_to_process:
                    break

        return lines

    def sequences_after_line(self, lineno, minlength=5):
        after = []
        for seq in sorted(self.increasing_sequences, key=operator.itemgetter(1)):
            if (lineno <= seq[1] or lineno <= seq[2]) and seq[0] > minlength:
                after.append(seq)

        return after

def main():

    document_citations = []

    for fname in sys.argv[1:]:
        print("Processing {}".format(os.path.basename(fname)))

        citation_group = CitationGroup(fname)

        document_citations.append(citation_group)

    for d in document_citations:
        if not d.increasing_sequences or not d.all_citations:
            print("no sequence/citations found for {}".format(d.name))
    for d in document_citations:
        print(d)
        print(d._get_references_text())
        print("--------------------------------------------------")

if __name__ == '__main__':
    main()
