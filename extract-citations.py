#!/usr/bin/env python

import sys
import os
import re
import string
import math
import operator
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class ReferenceType(Enum):
    SQUARE=1
    DOTTED=2
    NAMED=3
    TRIGRAPH=4

class ReferenceGroup(object):
    # Gets all references which are enclosed within square brackets
    square_re = re.compile("\[([0-9]+)\]")
    # Some papers have references in the form "1. " followed by the reference
    # text. The reference number is assumed to be below 1000, which filters out
    # possible conflicts with years in the reference text. Sometimes the reference
    # numbers are separated from the text, in which case they will be alone on a
    # line with the end of the line directly after the period. If not, a space
    # will separate the number and the reference text.
    dotted_re_string = "([0-9]{1,3})\.(\s+|$)"
    dotted_re = re.compile("^" + dotted_re_string)
    # need this to split a large string with regex
    dotted_re_nonstart = re.compile(dotted_re_string)

    # Some references in the text itself will be in the form of (surname et al.
    # 2001b; surname and other 1999), this captures those (these are quite rare
    # though)
    named_re = re.compile("\(((?:[a-zA-Z0-9 \n\.,]+(?:[0-9]{4}[a-z]*)+[; ]*)+)\)")
    # In the references section there is no easy way to determine where one
    # reference starts and another begins, seems like the year might be one?
    # Assume years are between 1900 and 2099
    named_ref_re = re.compile("(\(*(?:19|20)[0-9]{2}[a-z]*\)*)")
    # For some named ref documents, it can be more reliable to use the separator
    # for authors, e.g. Bariya, P., and Zheng, S.; and Lowe, D. The last part of
    # the or is necessary for single authors in the same scheme (sometimes)
    author_sep_re = re.compile("(?:\.,|\.;|,[ a-zA-Z]{,3}\.)")


    # The AMS authorship trigraph takes the form [FS+90, AB+91]. Assume that
    # they are usually comma separated if multiple references in same bracket
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
        self.max_reference_num = 0
        # Contains all found references, with 2-tuple containing reference number and line number
        self.square_references = []
        self.dotted_references = []
        self.named_references = []
        self.trigraph_references = []
        self.all_references = []
        self.increasing_sequences = []

        # This will contain references once the process is finished
        self.references = []

        # process the input file to extract some relevant bits
        self._process_file()

        # If we didn't get any references, then this document is probably one
        # with named references, so need to handle things differently
        if not self.dotted_references and not self.square_references and not self.trigraph_references:
            self._process_file_named_references()
            self.all_references = self.named_references
            self.reference_type = ReferenceType.NAMED
            self.compute_sequences(ignore_number=True)
        elif self.trigraph_references:
            # Trigraph and named references do not contain information about the
            # number of references which exist, so for consistency we must convert
            # the values we saw
            # Trigraph references are not grouped
            tri_set = set()
            for item in self.trigraph_references:
                tri_set.update(item[0].split(', '))

            self.max_reference_number = len(tri_set)
            self.all_references = self.trigraph_references
            self.reference_type = ReferenceType.TRIGRAPH
            self.compute_sequences(ignore_number=True)
        elif len(self.dotted_references) > max(self.max_reference_num/2, 10):
            # Dotted references are relatively rare, most papers use square brackets.
            # If they are used, then they only appear in the references section
            # Assume that any number of dotted references below half the total number
            # of references are just noise. Some papers with low max reference counts
            # might have a weird effect, so assume that if there aren't more than 10
            # dotted references that this is a paper with square references
            self.all_references = sorted(self.dotted_references + self.square_references, key=operator.itemgetter(1))
            self.reference_type = ReferenceType.DOTTED
            self.compute_sequences()
        else:
            self.all_references = self.square_references
            self.reference_type = ReferenceType.SQUARE
            self.compute_sequences()

        self._get_references_end()
        self._extract_references()

    def __str__(self):
        str_rep = ""
        str_rep += self.name + "\n"
        str_rep += "References start at {}, end at {}\n".format(self.references_start, self.references_end)
        str_rep += "Total references: {}\n".format(self.max_reference_num)
        str_rep += "Reference type seems to be {}\n".format(self.reference_type)
        str_rep += "Increasing sequences:\n {}\n".format(self.increasing_sequences)
        str_rep += "Square refs: {}, dotted refs: {}, named refs: {}, trigraph refs: {}\n".format(len(self.square_references), len(self.dotted_references), len(self.named_references), len(self.trigraph_references))

        return str_rep

    def _process_file(self):
        with open(self.fname) as f:
            for lineno, line in enumerate(f):
                lineno += 1 # enumeration starts at zero
                # crude method to extract the starting reference line, which works very well
                if "references" in line.lower() and len(line) < len("references") + 5:
                    self.references_start = lineno

                # try to get some information about possible supplementary
                # material after the references
                for end_string in self.end_strings:
                    # The supplementary sections usually have a title which is
                    # on its own line, so we might be able to ignore mentions in
                    # references or other parts of the paper where lines will be
                    # longer. This is usually only a problem with papers which
                    # use a named reference scheme
                    if end_string in line.lower() and len(line) < len(end_string) + 10:
                        self.end_material_lines.append(lineno)

                # Look for all square bracket references in a line, and add the
                # line number to the list for each one found
                for match in self.square_re.finditer(line):
                    reference_number = int(match.group(1))
                    self.square_references.append((reference_number, lineno))
                    # Also check the number of the reference so we can know how
                    # many references there are in the paper
                    if reference_number > self.max_reference_num:
                        self.max_reference_num = reference_number

                for match in self.trigraph_re.finditer(line):
                    self.trigraph_references.append((match.group(1), lineno))

                dotted = self.dotted_re.match(line)
                if dotted:
                    reference_number = int(dotted.group(1))
                    self.dotted_references.append((int(dotted.group(1)), lineno))
                    if reference_number > self.max_reference_num:
                        self.max_reference_num = reference_number

    def _process_file_named_references(self):
        with open(self.fname) as f:
            # First, extract the named references in the main text of the paper.
            # These are extractable, but we need to do more to get the actual
            # full references.
            tmp_refs = []
            for match in self.named_re.finditer(f.read()):
                # match.start gives you the character in the file not the line number
                tmp_refs.append((match.group(1), match.start()))

            # The initial extraction will extract ; separated references, want
            # to split them into their own ref
            split_refs = []
            for item in tmp_refs:
                split = item[0].replace("\n", ' ').split('; ')
                # If the split works, there is more than one ref in the group,
                # so create new tuples for each with the same line
                if len(split) > 1:
                    split_refs.extend([(sp, item[1]) for sp in split])
                else:
                    split_refs.append((split, item[1]))

            # need to go through the file again to get to the line where
            # refs start. We also count the cumulative number of characters
            # so that we can associate a character position with a specific
            # line
            file_linechars = []
            end_refs = []
            with open(self.fname) as f:
                for lineno, line in enumerate(f):
                    # need to get line length in bytes
                    linelength_bytes = sys.getsizeof(line)
                    file_linechars.append(linelength_bytes if lineno == 0 else file_linechars[lineno-1] + linelength_bytes)
                    if lineno >= self.references_start:
                        match = self.named_ref_re.search(line)
                        if match:
                            end_refs.append((line, lineno + 1))

            # This is more approximate than for other types of references
            self.max_reference_num = len(end_refs)
                            
            # Change the position in bytes to a position in lines for the
            # references in the main text (i.e. not end refs)
            line_refs = []
            for item in split_refs:
                # not super efficient
                for lineno, cm_linechars in enumerate(file_linechars):
                    if item[1] <= cm_linechars:
                        line_refs.append((item[0], lineno + 1)) # +1, first line is index 1, enum from 0
                        break

                    
            self.named_references = line_refs + end_refs
                        
    def compute_sequences(self, min_length=3, max_gap=8, ignore_number=False):
        """min_length is the minimum sequence length that will be considered

        max_gap is the maximum gap between two references. We want to use this
        function to extract references from the references section rather than
        the body of the paper, and those tend to be pretty close to each other.

        ignore_number will just use the gap and min length rather than looking
        at the number of the reference to check if it is increasing. This is
        useful for the trigraph type which isn't numbered

        """
        self.increasing_sequences = []
        if len(self.all_references) < 2:
            return

        # 3-tuple, sequence length, start line, end line
        start_ind = 0 # this starting index means that the first sequence is one shorter than it should be?
        start_line = self.all_references[0][1]

        for ind, item in enumerate(self.all_references[1:]):
            # indexing into the original list with ind will give us the previous
            # item, since we are slicing the enumerated list from 1.
            prev_item = self.all_references[ind]
            # Strictly less, because we really want to find sequences from the
            # references section which should be strictly increasing.
            if (ignore_number or prev_item[0] < item[0]) and item[1] - prev_item[1] < max_gap:
                if not start_ind:
                    start_ind = ind-1
                    start_line = prev_item[1]
            else:
                if start_ind and ind+1 - start_ind > min_length:
                    # references per line, might be useful
                    num_lines = ind - start_ind
                    if prev_item[1] == start_line: # the sequence ends on a single line
                        cpl = num_lines
                    else:
                        cpl = num_lines/float(prev_item[1]-start_line)
                    self.increasing_sequences.append((num_lines, start_line, prev_item[1], cpl))

                start_ind = None # reset start ind so we know the sequence ended

        # only if the last sequence starts with an element in the list before
        # the last one. start_ind is none if there is no sequence to be
        # finalised
        if not start_line == self.all_references[-1][1] and start_ind is not None:
            # -1 because len gives number of objects which starts from 1, we want inds
            num_lines = len(self.all_references) - 1 - start_ind
            # Add the final sequence
            cpl = num_lines/float(self.all_references[-1][1]-start_line)
            self.increasing_sequences.append((num_lines, start_line, self.all_references[-1][1], cpl))

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

    def _extract_references(self):
        text_lines = self._get_references_text()

        if not text_lines:
            return

        # Get the line numbers of references after the start of the reference
        # block, and make the start line of the reference block 0. -1 because
        # when constructing we start line numbers at 1 as with files, but
        # here we want array indices
        # reference_lines = [reference[1] - self.references_start - 1 for reference in sorted(self.all_references, key=operator.itemgetter(1)) if reference[1] >= self.references_start]

        # for ind, lineno in enumerate(reference_lines):
        #     end_ind = reference_lines[ind+1] if ind < len(reference_lines) - 1 else len(text_lines)
        #     print(text_lines[lineno:end_ind])

        # We already applied the regex to get values in all_references, but it's sensible to apply it again for simplicity
        if self.reference_type is ReferenceType.DOTTED:
            # use the nonstart version of the dotted regex, since the one used
            # before has ^ at the start, which will not work with a long string
            # rather than individual lines
            references = self.dotted_re_nonstart.split(text_lines)
        elif self.reference_type is ReferenceType.TRIGRAPH:
            references = self.trigraph_re.split(text_lines)
        elif self.reference_type is ReferenceType.SQUARE:
            references = self.square_re.split(text_lines)
        else:
            references = self._extract_references_named(text_lines)
                

        for reference in references:
            if len(reference) > 20: # some junk gets created by regex sometimes
                # hyphens directly before a newline indicate word continuation
                clean = reference.replace("-\n", '')
                # want to get the whole thing on one line
                self.references.append(clean.replace("\n", ' '))

    def _extract_references_named(self, text_lines, authsep_range=0.3):
        """Named reference extraction is a bit more involved because the reference style
        varies significantly

        authsep_range if the number of lines with author separators on them is
        within this percentage of the estimated maximum number of references, we
        assume that we should use author separators to group references

        """
        splitlines = text_lines.split("\n")
        # Check if it's better to use the author separators to get a good place
        # to join. Assume at if the number of lines with author separators is
        # quite close to the estimated total number of references then it's
        # sensible to use that to split instead of another method
        author_lines = []
        cur_ref_start = 0
        prev_line = 0
        for lineno, line in enumerate(splitlines):
            if self.author_sep_re.search(line):
                if not author_lines: # first line with names is always appended
                    cur_ref_start = lineno
                    author_lines.append(lineno)
                elif lineno - prev_line > 1:
                    # if not first line, the line number has to be at least 2
                    # bigger than the previous one. This squashes multiline author sections
                    author_lines.append(lineno)

                prev_line = lineno

        use_authsep = abs(len(author_lines) - self.max_reference_num) < authsep_range * self.max_reference_num

        if use_authsep:
            end_refs = author_lines
        else:
            end_refs = [r[1]-self.references_start for r in self.all_references if r[1] > self.references_start]

        refs = []
        for ind, ref in enumerate(end_refs[1:]):
            refs.append(" ".join(splitlines[end_refs[ind]:ref]))
            
        return refs

    def get_references_as_sets(self):
        """To match the references from various papers, it is useful to have them in a
        set representation so we can check the intersection. 
        """
        set_rep = []
        for ref in self.references:
            set_rep.append(set(ref.translate(None, ',.():').split(' ')))

        return set_rep
    
    def sequences_after_line(self, lineno, minlength=5):
        after = []
        for seq in sorted(self.increasing_sequences, key=operator.itemgetter(1)):
            if (lineno <= seq[1] or lineno <= seq[2]) and seq[0] > minlength:
                after.append(seq)

        return after

def main():

    document_references = []

    for fname in sys.argv[1:]:
        print("Processing {}".format(os.path.basename(fname)))

        reference_group = ReferenceGroup(fname)

        document_references.append(reference_group)

    for d in document_references:
        if not d.increasing_sequences or not d.all_references:
            print("no sequence/references found for {}".format(d.name))
    for d in document_references:
        if d.reference_type == ReferenceType.NAMED:
            print(d)
            print(d.get_references_as_sets())
            print("--------------------------------------------------")

if __name__ == '__main__':
    main()
