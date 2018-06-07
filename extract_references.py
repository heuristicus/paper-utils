#!/usr/bin/env python

import sys
import os
import re
import string
import math
import operator
import logging
import unidecode
from enum import Enum

def refgroup_from_filelist(filelist, filename_regex=None):
    """Given a list of files, will create a reference group from each and add it to
    a list, which is returned at the end.

    ignore_fnames allows the format of the file name to be ignored. 

    """
    docs = []
    for fname in sys.argv[1:]:
        if not os.path.isfile(fname):
            continue

        try:
            reference_group = ReferenceGroup(fname, filename_regex)
            docs.append(reference_group)
        except ValueError as ex:
            logging.getLogger().warning((str(ex)))

    return docs

class FilenameRegex(object):
    """Used as an optional parameter to a referencegroup to extract data from the
    filename

    """
    def __init__(self, regex=None, author_ind=None, title_ind=None, year_ind=None):
        self.regex = regex
        self.author_ind = author_ind
        self.title_ind = title_ind
        self.year_ind = year_ind

        if self.is_valid():
            self.regex = re.compile(self.regex)
            
    def is_valid(self):
        """For this object to be valid, must have a regex string and at least one of the
        author, title or year indices set

        """
        return self.regex is not None and (self.author_ind or self.title_ind or self.year_ind)

    def get_data(self, filename):
        """Extract the author, title, and year from the given filename, using the regex
        specified. Returns a tuple of (author, title, year), or None if no match
        was found or 

        """
        match = self.regex.match(filename)
        if match:
            author = match.group(self.author_ind) if self.author_ind else None
            title = match.group(self.title_ind) if self.title_ind else None
            year = match.group(self.year_ind) if self.year_ind else None
            return (author, title, year)
        else:
            return None
        
class ReferenceType(Enum):
    SQUARE=1
    DOTTED=2
    NAMED=3
    TRIGRAPH=4

class ReferenceGroup(object):
    # For the last reference in a document, we don't know exactly where it ends,
    # so we assume that we should consider it to run for this number of lines
    # after its start point.
    REF_END_PADDING = 3
    # If an extracted has less than this number of characters, then it's
    # probably not a reference
    MIN_REF_CHARS = 20
    # Assume that references are not any longer than this number of lines
    MAX_REF_LINES = 5

    # Gets all references which are enclosed within square brackets
    square_re = re.compile("\[([0-9]+)\]")
    # Some papers have references in the form "1. " followed by the reference
    # text. The reference number is assumed to be below 1000, which filters out
    # possible conflicts with years in the reference text. Sometimes the
    # reference numbers are separated from the text, in which case they will be
    # alone on a line with the end of the line directly after the period. If
    # not, a space will separate the number and the reference text. Use
    # multiline because we want to use this to split a text block later
    dotted_re = re.compile("^([0-9]{1,3})\.(\s+|$)", re.MULTILINE)

    # Some references in the text itself will be in the form of (surname et al.
    # 2001b; surname and other 1999), this captures those (these are quite rare
    # though)

    # (?:\)|\])* section after the year causes the expression to break out of the brackets... need to fix
    named_re = re.compile("(?:\(|\[)((?:[ a-zA-Z\.,\n-]+(?:\(|\[)*(?:19|20)[0-9]{2}(?:\)|\])*[; \n]*)+)(?:\)|\])")
    # In the references section there is no easy way to determine where one
    # reference starts and another begins, seems like the year might be one?
    # Assume years are between 1900 and 2099
    named_ref_re = re.compile("\(*((?:19|20)[0-9]{2}[a-z]*)\)*")
    # For some named ref documents, it can be more reliable to use the separator
    # for authors, e.g. Bariya, P., and Zheng, S.; and Lowe, D. The third part
    # of the or is necessary for single authors in the same scheme (sometimes).
    # The next part is useful for extracting authors specifically in the
    # references section, because title names rarely have commas in them. The
    # next one extracts author initials where commas are not used
    author_sep_re = re.compile("(?:[A-Z]\.,|[A-Z]\.;|, [A-Z]\.|[a-zA-Z\n], | [A-Z] )")


    # The AMS authorship trigraph takes the form [FS+90, AB+91]. Assume that
    # they are usually comma separated if multiple references in same bracket
    # pair
    trigraph_re = re.compile("\[((?:[A-Z+]+[0-9]{2}[, ]*)+)\]")

    # Some papers have end material which we should make sure to ignore when
    # looking at the references, these strings often appear at the start of
    # those sections.
    end_strings = ["appendix", "appendices", "supplementary"]
    start_strings = ["bibliography", "references", "citations"]

    group_id = 0

    def __init__(self, fname, filename_regex=FilenameRegex()):
        """fname is the file name to read from

        filename_regex expects a FilenameRegex object which specifies a regular
        expression and additional information to use to extract the author,
        title and year from the file name. If we wanted to extract from files
        with the format "Author - year - title", then we would pass the object
        FilenameRegex("(.*) - (.*) - (.*)", author_ind=1, year_ind=2,
        title_ind=3). Each group (i.e. parenthesis enclosed section) will be
        assumed to contain one of the three bits of information we want to
        extract. The latter parameters define which group contains which piece
        of information

        """
        self.group_id = ReferenceGroup.group_id
        ReferenceGroup.group_id += 1
        
        base_file = os.path.splitext(os.path.basename(fname))[0]
        if filename_regex and filename_regex.is_valid():
            data = filename_regex.get_data(base_file)
            if data:
                self.author, self.title, self.year = data
                self.author = str(unidecode.unidecode(unicode(self.author, 'utf8')))
                self.title = str(unidecode.unidecode(unicode(self.title, 'utf8')))
            else:
                raise ValueError('Filename "{}" does not conform to regex "{}" that was provided'.format(base_file, filename_regex.regex.pattern))
        else:
            self.title = base_file

        self.logger = logging.getLogger()

        self.fname = fname
        # shortened name based on author, year and title, or if author and year
        # not available, just title
        self._short_name = None
        # guess about where the references section starts
        self.references_start = 0
        # guess about where the references section ends
        self.references_end = 0
        # how many lines ended up being processed from the references sections
        self.lines_processed = 0
        # lines on which "start strings" appear
        self.ref_start_lines = []
        # lines on which "end strings" appear
        self.end_material_lines = []
        # the highest number extracted from common citation patterns, assume
        # that the highest number is the number of total citations
        self.max_reference_num = 0
        # tuples of number, line for square-bracket references
        self.square_references = []
        # tuples of number, line for number and point references (e.g. 32.)
        self.dotted_references = []
        # tuples of line, line number for non-numbered references
        self.named_references = []
        # tuples of citation text (e.g. XYZ+90), line number for trigraph references
        self.trigraph_references = []
        # Contains all found references, with 2-tuple containing reference number/content and line number
        self.all_references = []
        # tuples of (number of references, start line, end line, references per
        # line). Each tuple represents an increasing sequence of citation
        # numbers in the given line range, if the citations are numbered.
        # Otherwise it will consider citations which are close to each other in
        # the text
        self.sequences = []
        # This will contain reference text once the process is finished
        self.references = []

        # process the input file to do initial regex extraction
        self._process_file()

        self.all_references = self.square_references + self.dotted_references + self.trigraph_references

        # Compute sequences where the reference number is increasing (if the references are numbered),
        self.compute_sequences()

        self._get_references_start()
        self._get_references_end()

        # Check that the reference numbers are valid by making sure they are in
        # a valid range
        self.reference_nums_valid = self._valid_reference_nums()

        if not self.reference_nums_valid or not self.sequences:
            # If we didn't manage to get any sequences, or the references were
            # invalid, then this document is probably one with named references,
            # so need to handle things differently
            self._process_file_named_references()
            self.all_references = self.named_references
            self.reference_type = ReferenceType.NAMED
            self.compute_sequences()
            self._get_references_start()
            self._get_references_end()
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
        else:
            # Deduce the overall reference type for the document by looking at
            # sequences. Sometimes documents which use square refs in the main
            # body use dotted refs in the reference section, so we look at the
            # refs in that section to decide

            square = [r for r in self.square_references if self.references_start < r[1] < self.references_end]
            dotted = [r for r in self.dotted_references if self.references_start < r[1] < self.references_end]

            self.reference_type = ReferenceType.SQUARE if len(square) > len(dotted) else ReferenceType.DOTTED

            if self.reference_type == ReferenceType.DOTTED:
                # Dotted references are relatively rare, most papers use square
                # brackets. If they are used, then they only appear in the
                # references section Assume that any number of dotted references
                # below half the total number of references are just noise. Some
                # papers with low max reference counts might have a weird
                # effect, so assume that if there aren't more than 10 dotted
                # references that this is a paper with square references
                self.all_references = sorted(self.dotted_references + self.square_references, key=operator.itemgetter(1))
            else:
                self.all_references = self.square_references

        self._extract_references()

    def __str__(self):
        str_rep = ""
        str_rep += self.title + "\n"
        str_rep += "References start at {}, end at {}\n".format(self.references_start, self.references_start + self.lines_processed)
        str_rep += "Total references: {}\n".format(len(self.references))
        str_rep += "Reference type seems to be {}\n".format(self.reference_type)
        str_rep += "Sequences: {}\n".format(sorted(self.sequences, key=operator.itemgetter(0), reverse=True))
        str_rep += "Square refs: {}, dotted refs: {}, named refs: {}, trigraph refs: {}\n".format(len(self.square_references), len(self.dotted_references), len(self.named_references), len(self.trigraph_references))

        return str_rep

    def short_name(self):
        if not self._short_name:
            if self.author and self.year:
                self._short_name =  "{} ({}) {}".format(self.author, self.year, self.title.split(" ")[0])
            else:
                self._short_name = " ".join(self.title.split(" ")[:3])

        return self._short_name
    
    def _process_file(self):
        with open(self.fname) as f:
            for lineno, line in enumerate(f):
                lineno += 1 # enumeration starts at zero
                # crude method to extract the starting reference line, which works very well
                for start_string in self.start_strings:
                    if start_string in line.lower() and len(line) < len(start_string) + self.REF_END_PADDING:
                        self.ref_start_lines.append(lineno)

                # try to get some information about possible supplementary
                # material after the references
                for end_string in self.end_strings:
                    # The supplementary sections usually have a title which is
                    # on its own line, so we might be able to ignore mentions in
                    # references or other parts of the paper where lines will be
                    # longer. This is usually only a problem with papers which
                    # use a named reference scheme
                    if end_string in line.lower():
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

                # These can cause some issues because numbers appear in the
                # text, or in references as page numbers. The
                # valid_reference_nums function should do something about this
                dotted = self.dotted_re.match(line)
                if dotted:
                    reference_number = int(dotted.group(1))
                    self.dotted_references.append((int(dotted.group(1)), lineno))
                    if reference_number > self.max_reference_num:
                        self.max_reference_num = reference_number

    def _process_file_named_references(self):

        author_split_re = re.compile("; *")

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
                split = author_split_re.split(item[0].replace("\n", ' '))
                # If the split works, there is more than one ref in the group,
                # so create new tuples for each with the same line
                if len(split) > 1:
                    split_refs.extend([(sp, item[1]) for sp in split])
                else:
                    split_refs.append((split[0], item[1]))

            # need to go through the file again to get to the line where
            # refs start. We also count the cumulative number of characters
            # so that we can associate a character position with a specific
            # line.
            file_linechars = []
            end_refs = []
            end_years = []
            with open(self.fname) as f:
                for lineno, line in enumerate(f):
                    # need to get line length in bytes
                    linelength_bytes = sys.getsizeof(line)
                    file_linechars.append(linelength_bytes if lineno == 0 else file_linechars[lineno-1] + linelength_bytes)
                    if lineno >= self.references_start:
                        match = self.named_ref_re.search(line)
                        if match:
                            end_refs.append((line, lineno + 1))
                            end_years.append(match.group(1))

            # The end references regex can be bad for separating references,
            # especially if there are multiple years in a single citation in bad
            # cases. Because of this, go over the list and look at the years
            # extracted, keeping only one of a pair of years, if the line
            # numbers of the pair are close
            pruned_end_refs = []
            ind = 0
            while ind < len(end_refs) - 1:
                years_equal = end_years[ind] == end_years[ind+1]
                line_dist = abs(end_refs[ind][1] - end_refs[ind+1][1])

                pruned_end_refs.append(end_refs[ind])

                if not years_equal or line_dist > 2:
                    ind += 1
                else:
                    ind += 2 # skip over the next item

            # This is more approximate than for other types of references
            self.max_reference_num = len(pruned_end_refs)

            # Change the position in bytes to a position in lines for the
            # references in the main text (i.e. not end refs)
            line_refs = []
            for item in split_refs:
                # not super efficient
                for lineno, cm_linechars in enumerate(file_linechars):
                    if item[1] <= cm_linechars:
                        line_refs.append((item[0], lineno + 1)) # +1, first line is index 1, enum from 0
                        break

            self.named_references = line_refs + pruned_end_refs

    def _valid_reference_nums(self, invalid_prop=0.1, number_cutoff=500):
        """Check the numbers retrieved from the extraction of reference data to make
        sure they are as we would expect of a paper. i.e. they start at 1, and
        go up to some other number with no gaps. If this is not the case, then
        the paper is most likely using named references. Only checks reference
        numbers after the assumed start of the references

        invalid_prop if invalid/valid > invalid_prop, then consider the
        references to be invalid

        number_cutoff any reference above this number is considered to be invalid

        """
        # Get all the references in ascending order, ignoring zeros
        allrefs_tmp = self.dotted_references + self.square_references
        all_refnums = filter(lambda x:x!=0, sorted([r[0] for r in allrefs_tmp if r[1] >= self.references_start]))
        if not all_refnums:
            self.max_reference_num = None
            return False

        if all_refnums[0] != 1:
            # The first reference number should always be 1
            self.max_reference_num = None
            return False

        # The maximum spacing between refnums should be 1. However, it is common
        # that there are some random incorrect values scattered in the extracted
        # data because of the nature of things, so we have to do another check
        # at the end to see if this is systematic or just a blip
        valid_refnums = 0
        invalid_refnums = 0
        for ind, r in enumerate(all_refnums[1:]):
            if abs(all_refnums[ind] - r) > 1 or all_refnums[ind] > number_cutoff:
                invalid_refnums += 1
            else:
                valid_refnums += 1

            # assume that any reference above 100 is a year
        if invalid_refnums / float(valid_refnums) > invalid_prop:
            self.max_reference_num = None
            return False

        return True

    def compute_sequences(self, min_length=3, max_line_gap=8, max_num_gap=1):
        """min_length is the minimum sequence length that will be considered

        max_line_gap is the maximum gap between two references. We want to use this
        function to extract references from the references section rather than
        the body of the paper, and those tend to be pretty close to each other.

        max_num_gap is the maximum difference between the citation number to
        consider it an increasing sequence. By default it is 1, to indicate that
        we want to find increasing sequences which look like bibliographies

        """
        self.sequences = []
        if len(self.all_references) < 2:
            return

        # If this is true, which is is for types other than dotted and square,
        # we do not consider the citation number in the sequence. This means
        # that it's less reliable in terms of getting correct values, but is
        # still useful for determining the start and end of the citations
        if isinstance(self.all_references[0][0], str):
            ignore_number = True
        else:
            ignore_number = False

        # 3-tuple, sequence length, start line, end line
        start_ind = 0 # this starting index means that the first sequence is one shorter than it should be?
        start_line = self.all_references[0][1]

        for ind, item in enumerate(self.all_references[1:]):
            # indexing into the original list with ind will give us the previous
            # item, since we are slicing the enumerated list from 1.
            prev_item = self.all_references[ind]
            # Strictly less, because we really want to find sequences from the
            # references section which should be strictly increasing.
            line_gap_ok = item[1] - prev_item[1] < max_line_gap
            if not ignore_number:
                num_gap_ok = item[0] > prev_item[0] and abs(item[0] - prev_item[0]) <= max_num_gap
            if (ignore_number or num_gap_ok) and line_gap_ok:
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
                    self.sequences.append((num_lines, start_line, prev_item[1], cpl))

                start_ind = None # reset start ind so we know the sequence ended
            
        # only if the last sequence starts with an element in the list before
        # the last one. start_ind is none if there is no sequence to be
        # finalised
        if not start_line == self.all_references[-1][1] and start_ind is not None:
            # -1 because len gives number of objects which starts from 1, we want inds
            num_lines = len(self.all_references) - 1 - start_ind
            # Add the final sequence
            cpl = num_lines/float(self.all_references[-1][1]-start_line)
            self.sequences.append((num_lines, start_line, self.all_references[-1][1], cpl))

    def _get_references_start(self):
        # If there is only one reference start line, then set it as the starting point
        if len(self.ref_start_lines) == 1:
            self.references_start = self.ref_start_lines[0]
        elif len(self.ref_start_lines) > 1:
            # Otherwise, Use the extracted sequences to determine a starting point
            # Will go through each of the ref start lines and all the sequences,
            # and see how far away each of the lines is from the beginning of a
            # sequence. Will choose start based on the smallest gap. If there
            # are two equal gaps, then... pick the one with the most references.
            # Some papers have multiple reference sections for the main paper
            # and appendices
            min_gaps = []
            for ref_line in self.ref_start_lines:
                min_gap = None
                min_ind = None
                min_line = ref_line
                for ind, seq in enumerate(self.sequences):
                    gap = abs(seq[1]-ref_line)
                    if not min_gap or gap < min_gap:
                        min_gap = gap
                        min_ind = ind

                min_gaps.append((min_ind, min_gap, min_line))

            # smallest gap first
            min_gaps = sorted(min_gaps, key=operator.itemgetter(1))
            # have all gaps now, need to check if there are multiple gaps with
            # the same minimum distance, which can happen if there are multiple reference sections
            multiple = [min_gaps[0]]
            for ind, gap_t in enumerate(min_gaps[1:]):
                if gap_t[1] == min_gaps[ind][1]:
                    multiple.append(gap_t)

            # If we get several minimum gaps of the same length, then we take
            # the start to be the one which is closest to the sequence with the
            # largest number of references
            if multiple:
                max_refs = 0
                max_ind = 0
                for ind, gap_t in enumerate(multiple):
                    # gap_t stores the sequence index
                    seq = self.sequences[gap_t[0]]
                    if seq[0] > max_refs:
                        max_refs = seq[0]
                        max_ind = ind
                self.references_start = multiple[max_ind][2]
            else:
                self.references_start = min_gaps[0][2]

        else:
            self.references_start = sorted(self.sequences, key=operator.itemgetter(1))[0][1]

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

            if self.end_material_lines and self.references_end:
                self.logger.debug("Got end material lines and end of refs")
                # both end material and estimated ref end from the ref sequence
                # exists, need to combine the two. If end material lines only
                # happen before the start of references, then ignore them
                last_end_material = max(self.end_material_lines)
                end_mat_in_seq = False
                # Check if the end material appears in the middle of a sequence,
                # and if it does, it is discarded - sequences are stronger than
                # end material
                for seq in self.sequences:
                    if seq[1] <= last_end_material <= seq[2]:
                        end_mat_in_seq = True
                        break
                    
                if last_end_material < self.references_start or end_mat_in_seq:
                    self.logger.debug("End material lines end at {}, references start at {}".format(last_end_material, self.references_start))
                    lines_to_process = self.references_end - self.references_start + self.REF_END_PADDING
                else:
                    self.logger.debug("End material exists (line {}) after the start of references (line {})".format(last_end_material, self.references_start))
                    
                    lines_to_process = min(max(self.end_material_lines), self.references_end) - self.references_start + self.REF_END_PADDING
            elif self.end_material_lines:
                self.logger.debug("Got only end material lines")
                if max(self.end_material_lines) > self.references_start:
                    # If there is an appendix or supplementary material, that
                    # usually comes directly after the references section, so
                    # cutoff can be there.
                    lines_to_process = max(self.end_material_lines) - self.references_start
                else:
                    # can't know where to end so have to process everything
                    # after the references start
                    lines_to_process = None
            elif self.references_end:
                self.logger.debug("got only references end")
                # add a bit of padding so that we get the text of the last reference
                lines_to_process = self.references_end - self.references_start + self.REF_END_PADDING
            else:
                self.logger.debug("got no information about end material or references end")
                lines_to_process = None

            self.lines_processed = 0
            for line in f:
                lines += line
                self.lines_processed += 1
                if lines_to_process and self.lines_processed >= lines_to_process:
                    break

        return lines

    def _extract_references(self):
        text_lines = self._get_references_text()

        if not text_lines:
            return

        # We already applied the regex to get values in all_references, but it's sensible to apply it again for simplicity
        if self.reference_type is ReferenceType.DOTTED:
            # use the nonstart version of the dotted regex, since the one used
            # before has ^ at the start, which will not work with a long string
            # rather than individual lines
            references = self.dotted_re.split(text_lines)
        elif self.reference_type is ReferenceType.TRIGRAPH:
            references = self.trigraph_re.split(text_lines)
        elif self.reference_type is ReferenceType.SQUARE:
            references = self.square_re.split(text_lines)
        else:
            references = self._extract_references_named(text_lines)


        for reference in references:
            # Don't add a "reference" which is less than this number of chars, it's unlikely to be one
            if len(reference) > self.MIN_REF_CHARS: # some junk gets created by regex sometimes
                # Truncate super long references, which can happen when text and references are mixed
                if reference.count("\n") > self.MAX_REF_LINES:
                    reference = ''.join(reference.split("\n")[:self.MAX_REF_LINES])
                # hyphens directly before a newline indicate word continuation
                # Also want to get the whole thing on one line
                clean = reference.replace("-\n", '').replace("\n", ' ')
                self.references.append(clean)

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

        end_refs.append(end_refs[-1] + self.REF_END_PADDING)

        refs = []
        for ind, ref in enumerate(end_refs[1:]):
            refs.append("\n".join(splitlines[end_refs[ind]:ref]))

        return refs

    def get_references_as_sets(self):
        """To match the references from various papers, it is useful to have them in a
        set representation so we can check the intersection.
        """
        set_rep = []
        for ref in self.references:
            set_rep.append(set(ref.translate(None, ',.():;').split(' ')))

        return set_rep

    def sequences_after_line(self, lineno, minlength=5):
        after = []
        for seq in sorted(self.sequences, key=operator.itemgetter(1)):
            if (lineno <= seq[1] or lineno <= seq[2]) and seq[0] > minlength:
                after.append(seq)

        return after

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
        
    document_references = refgroup_from_filelist(sys.argv[1:])

    for d in document_references:
        if not d.sequences or not d.all_references:
            logger.info("no sequence/references found for {}".format(d.name))
    logger.info("Final")
    for d in document_references:
        logger.info(d)
        for i, ref in enumerate(d.references):
            logger.info("{}: {}".format(i+1, ref))
        logger.info("--------------------------------------------------")

if __name__ == '__main__':
    main()
