#!/usr/bin/env python
import sys
import os
import logging
import argparse
import string
import pickle
from extract_references import ReferenceGroup, refgroup_from_filelist, FilenameRegex
import unidecode
import pygraphviz as pgv
from graph_tool.all import *

# Most usual stopwords would be quite unique if they appeared in a title, we are
# only interested in those which are really common or uninformative
stopwords = ['an', 'for', 'do', 'its', 'of', 'is', 'or', 'who', 'from', 'the', 'are', 'to', 'at', 'and', 'in', 'on', 'a', 'by']
# Some terms appear a lot in papers, might want to filter them
term_stopwords = ['convolutional', 'neural', 'networks', 'semantic', 'segmentation', 'deep']

# Strings which often come in the journal section of a reference
conf_strings = ["proceedings", "conference", "journal", "transactions", "letters", "advances", "arxiv", " conf.", " proc.", "ieee", "international", "int.", " volume ", "eccv", "icra", "cvpr", "iccv", " acm ", " in: ", "editors", " eds.", "ijcv", "tech report", " plos ", "isprs", "annals", "springer", "elsevier", " vol. ", " ed. ", " ch. ", "macmillan", "dissertation", "technical report", "mcgraw-hill", "workshop", "aaai"]


punctuation = ",.():;'\""
logger = logging.getLogger()

def match_reference_to_title(docs):
    edges = []

    graph = Graph()
    graph.add_vertex(len(docs))

    doc_sets = {d.short_name(): {} for d in docs}

    for from_ind, d in enumerate(docs):
        logger.debug("--------------------")
        logger.debug("checking {}, {}".format(d.author, d.title))
        for to_ind, d2 in enumerate(docs):
            if from_ind == to_ind:
                continue

            if "trimmed" not in doc_sets[d2.short_name()]:
                d2title_trimmed = d2.title.lower().translate(None, punctuation)
                d2title_stopped = [word for word in d2title_trimmed.split(" ") if word not in stopwords]
                d2title_set = set(d2title_stopped)
                doc_sets[d2.short_name()]["trimmed"] = d2title_trimmed
                doc_sets[d2.short_name()]["set"] = d2title_set
            else:
                d2title_trimmed = doc_sets[d2.short_name()]["trimmed"]
                d2title_set = doc_sets[d2.short_name()]["set"]

            for ref_ind, ref in enumerate(d.references):
                if "ref_sets" not in doc_sets[d.short_name()]:
                    doc_sets[d.short_name()]["ref_sets"] = []
                    doc_sets[d.short_name()]["ref_trims"] = []
                if len(doc_sets[d.short_name()]["ref_sets"]) <= ref_ind or len(doc_sets[d.short_name()]["ref_sets"]) == 0:
                    ref_trimmed = ref.lower().translate(None, punctuation)
                    ref_stopped = [word for word in ref_trimmed.split(" ") if word not in stopwords]
                    ref_set = set(ref_stopped)
                    doc_sets[d.short_name()]["ref_sets"].append(ref_set)
                    doc_sets[d.short_name()]["ref_trims"].append(ref_trimmed)
                else:
                    ref_set = doc_sets[d.short_name()]["ref_sets"][ref_ind]
                    ref_trimmed = doc_sets[d.short_name()]["ref_trims"][ref_ind]

                # lowercase string due to possible different capitalisation schemes, remove certain punctuation
                intersect = ref_set & d2title_set
                if len(intersect) > 5:
                    #logger.debug((intersect, d2title_stopped))
                    #logger.debug((ref, d2.title))
                    pass
                if d2title_trimmed in ref_trimmed:
                    #logger.debug("{}, {}".format(d2.title, ref))
                    # edge is directed from d->d2, since d has d2 in its references
                    #logger.debug("Adding edge between {} and {}".format(d.short_name(), d2.short_name()))
                    edges.append((d.short_name(), d2.short_name()))
                    graph.add_edge(graph.vertex(from_ind),graph.vertex(to_ind))

    #graph tool
    arf = arf_layout(graph, d=3, max_iter=5000)
    graph_draw(graph, pos=arf, output="arf.pdf")

    # pygraphviz
    # graph = pgv.AGraph(directed=True)
    # graph.add_edges_from(edges)
    # graph.write("refs.dot")
    # for node in graph.nodes_iter():
    #     print(node, graph.in_degree(node))

    # graph.draw("refs.pdf", prog="neato", args="-Goverlap=false")

#    logger.debug(sorted(graph.in_degree(with_labels=True).items(), key=operator.itemgetter(1)))

def similar_references(docs, overlap_prop = 0.6, min_set_size = 4):
    # If the intersection of the two sets is greater than this proportion of the
    # length of the shorter set, the two references are considered similar

    all_refs = []

    for d in docs:
        for r in d.references:
            stripped = r.lower().strip()
            stripped = str(unidecode.unidecode(unicode(stripped, 'utf8')))

            sep_inds, sep_inds_rev = punctuation_density_separate(stripped)
            if not sep_inds:
                continue

            punc_sep = stripped[sep_inds[0]:sep_inds_rev[-1]].strip()

            conf_inds = []
            for conf_str in conf_strings:
                found_ind = punc_sep.find(conf_str)
                # print("found conf string {} at ind {}".format(conf_str, found_ind))
                if found_ind > 0:
                    conf_inds.append(found_ind)

            logger.debug(conf_inds)
            if not conf_inds:
                # print("found no conference strings")
                conf_strip = punc_sep
            else:
                str_start = min(conf_inds)
                last_dot = punc_sep.rfind(".", 0, str_start)
                last_comma = punc_sep.rfind(",", 0, str_start)

                # print("conf string started at {}, last dot at {}, last comma at {}".format(str_start, last_dot, last_comma))
                # print("all after latest punctuation: {}".format(punc_sep[max(last_dot, last_comma):]))
                conf_strip = punc_sep[:max(last_dot, last_comma)]

            full_strip = conf_strip

            # print("conf_strip: {}\nfull strip: {}".format(conf_strip, full_strip))

            ref_trimmed = full_strip.lower().translate(None, punctuation)
            ref_stopped = [word for word in ref_trimmed.split(" ") if word not in stopwords and len(word) > 2]
            ref_set = set(ref_stopped)
            all_refs.append((r.strip(), ref_stopped, ref_set, d.short_name(), punc_sep, full_strip))



    # List for each reference of references which are similar
    similar = [[] for i in range(0, len(all_refs))]
    for cur_ind, current in enumerate(all_refs):
        logger.debug("{}/{}".format(cur_ind, len(all_refs)))
        logger.debug("--------------------------------------------------")
        logger.debug(all_refs[cur_ind][0])
        cur_setlen = len(current[2])
        for other_ind, other in enumerate(all_refs):
            other_setlen = len(other[2])
            if cur_ind >= other_ind:
                continue

            if cur_setlen <= min_set_size or other_setlen <= min_set_size:
                continue

            max_count = min(cur_setlen, other_setlen)
            common = current[2].intersection(other[2])
            overlap = len(common)/float(max_count)

            if overlap > overlap_prop:
                # Look at the stopped lists of words extracted from the
                # reference. This is quite a strict measure, doesn't consider
                # possibility of stuff after the title being included
                c_list = all_refs[cur_ind][1]
                o_list = all_refs[other_ind][1]
                if arrays_contain_same_reference(c_list, o_list, common):
                    similar[cur_ind].append(other_ind)
                    similar[other_ind].append(cur_ind)

                ref_same_slide(current[0], other[0])

    return all_refs, similar

def arrays_contain_same_reference(first, second, common, sequence_skip=2):
    """sequence_skip defines how far apart two elements of the sequence can be for
    them to still be considered in a sequence. e.g. with skip of 2, 1,2,4,5 is a
    valid sequence but 1,2,5,6 is not

    """

    if first == second:
        return True

    logger.debug("\n--------------------------------------------------")
    logger.debug("Arrays not equal:")
    logger.debug(first)
    logger.debug(second)
    logger.debug(common)
    # Find the index of occurrences of common words in the two references,
    # and use that information to get approximate extents of the title. If
    # the title is the same, then the length of the list from earliest to
    # latest occurrence of common words should be the same for both of the
    # lists given
    first_inds = []
    second_inds = []
    for word in common:
        first_inds.append(first.index(word))
        second_inds.append(second.index(word))

    first_inds.sort()
    second_inds.sort()
    logger.debug("inds common first: {}".format(first_inds))
    logger.debug("inds common second: {}".format(second_inds))

    first_min = first_inds[0]
    first_max = first_inds[-1]
    second_min = second_inds[0]
    second_max = second_inds[-1]

    extent_first = first[first_min:first_max + 1]
    extent_second = second[second_min:second_max + 1]
    logger.debug("common extent list first: {}".format(extent_first))
    logger.debug("common extent list second: {}".format(extent_second))

    if extent_first == extent_second:
        logger.debug("extents were the same")
        return True

    # If the extents are not equal, we should look at the indices and check to
    # see if the extracted words are in a sequence or not, which should indicate
    # where the title is and where there are words which we can consider to be
    # clutter (i.e. in journal titles)
    seq_first = [val for ind,val in enumerate(first_inds[:-1]) if abs(val - first_inds[ind+1]) <= sequence_skip]
    seq_second = [val for ind,val in enumerate(second_inds[:-1]) if abs(val - second_inds[ind+1]) <= sequence_skip]
    logger.debug(seq_first)
    logger.debug(seq_second)

    trimmed_extent_first = first[seq_first[0]:seq_first[-1]]
    trimmed_extent_second = second[seq_second[0]:seq_second[-1]]

    logger.debug(trimmed_extent_first)
    logger.debug(trimmed_extent_second)

    # Maybe we got rid of some of the confounding stuff?
    if trimmed_extent_first == trimmed_extent_second:
        logger.debug("trimmed extents were the same")
        return True

    return False

def similar_refs_slide(docs):
    # If the intersection of the two sets is greater than this proportion of the
    # length of the shorter set, the two references are considered similar

    all_refs = []

    for d in docs:
        for r in d.references:
            logger.debug("==================================================")
            logger.debug("'{}'".format(r))
            stripped = r.lower().strip()
            stripped = str(unidecode.unidecode(unicode(stripped, 'utf8')))

            conf_inds = []
            for conf_str in conf_strings:
                found_ind = stripped.find(conf_str)

                if found_ind > 0:
                    logger.debug("found conf string {} at ind {}".format(conf_str, found_ind))
                    conf_inds.append(found_ind)

            if not conf_inds:
                logger.debug("found no conference strings")
                conf_strip = stripped
            else:
                str_start = min(conf_inds)
                last_dot = stripped.rfind(".", 0, str_start) + 1
                last_comma = stripped.rfind(",", 0, str_start) + 1

                conf_strip = stripped[:max(last_dot, last_comma)]
                logger.debug("conf string started at {}, last dot at {}, last comma at {}".format(str_start, last_dot, last_comma))
                logger.debug("all after latest punctuation: {}".format(conf_strip))

            sep_inds, sep_inds_rev = punctuation_density_separate(conf_strip)
            if not sep_inds: # if it's empty this is usually just a spurious reference
                continue

            # If there are two or fewer separation indices then the separation
            # wasn't able to get good separation for authors, title and journal,
            # so we don't change anything
            if len(sep_inds) >= 2:
                # if there are 3 or more sep inds, we want to get all the text in the middle part
                punc_sep = conf_strip[sep_inds[0]:sep_inds_rev[-1]].strip()
            else:
                punc_sep = conf_strip

            logger.debug("conf_strip: {}\npunc_sep: {}\n".format(conf_strip, punc_sep))

            full_strip = punc_sep

            ref_stopped = [word for word in full_strip.split(" ") if word not in stopwords and len(word) > 2]
            ref_nopunc = " ".join(ref_stopped).translate(None, punctuation).split(" ")
            ref_set = set(ref_stopped)
            all_refs.append((r.strip(), ref_stopped, ref_nopunc, ref_set))

    # List for each reference of references which are similar
    similar = [[] for i in range(0, len(all_refs))]
    for cur_ind, current in enumerate(all_refs):
        logger.debug("{}/{}".format(cur_ind, len(all_refs)))
        logger.debug("--------------------------------------------------")
        logger.debug(all_refs[cur_ind][0])
        cur_setlen = len(current[3])
        for other_ind, other in enumerate(all_refs):
            other_setlen = len(other[3])

            if cur_ind >= other_ind:
                continue

            if cur_setlen == 0 or other_setlen == 0:
                continue

            max_count = min(cur_setlen, other_setlen)
            common = current[3].intersection(other[3])
            overlap = len(common)/float(max_count)

            if overlap > 0.5:
                # Look at the stopped lists of words extracted from the
                # reference. This is quite a strict measure, doesn't consider
                # possibility of stuff after the title being included
                if ref_same_slide(current, other):
                    similar[cur_ind].append(other_ind)
                    similar[other_ind].append(cur_ind)


    return all_refs, similar

def ref_same_slide(first_ref, second_ref):
    logger.debug("================================================== SLIDE")
    # use the stripped strings with no punctuation to do the sliding comparison
    first = first_ref[2]
    second = second_ref[2]

    logger.debug(first)
    logger.debug(second)

    if first == second:
        logger.debug("Stopped refs without punctuation were the same")
        return True

    scores = []
    longest_run = 0
    # slide the first array along the second one to see if there is a
    # significant number of elements which are the same at some point
    for i in range(1, len(first) + len(second)):
        # This is how much should be trimmed from each list. Required when the
        # shorter list goes past the end of the longer list. The minimum part
        # gives you how far beyond the end of the longest list the current index
        # is. The value of i is basically where the last element of the first
        # list sits in terms of the second. If i-len is negative, then we are
        # still on a valid index of the second list. If it is positive, then the
        # end of the first list has gone past the end of the second list. The
        # trim will remove elements from the first list which have gone past the
        # end, and remove elements correspondingly from the start of the other
        # list to keep the list length the same
        trim = max(min(i-len(first), i-len(second)), 0)
        # logger.debug("len first -i:{}".format(len(first) - i))
        # logger.debug("len second -i:{}".format(len(second) - i))
        # logger.debug("trimming {} from both".format(trim))
        first_slice_end = -trim if trim > 0 else 0
        second_slice_start = trim if trim > 0 else 0

        first_sliced = first[-i:first_slice_end or None]
        second_sliced = second[second_slice_start or None:i]

        # Also need to consider overlap. If one list is longer than the other,
        # then there will be a point like the following:
        #
        # -12345
        # 123456
        #
        # The second list is one longer than the first, with - indicating that
        # there is no overlap of the lists there because the first list does not
        # have enough elements. So, we need to remove the first element of the
        # second list to get lists of equal length

        if len(first_sliced) > len(second_sliced):
            first_sliced = first_sliced[:-abs(len(first_sliced) - len(second_sliced))]
        else:
            second_sliced = second_sliced[abs(len(first_sliced) - len(second_sliced)):]

        if first_sliced == second_sliced:
            logger.debug("ARRAYS IDENTICAL")
            print(first_sliced)
            print(second_sliced)

        # logger.debug("first slice len: {}, second slice len: {}".format(len(first_sliced), len(second_sliced)))
        # logger.debug("first: {}".format(first_sliced))
        # logger.debug("second: {}".format(second_sliced))

        score = 0
        match_inds = []
        run = 0
        for i, v in enumerate(first_sliced):
            if first_sliced[i] == second_sliced[i] and first_sliced[i] not in term_stopwords:
                score += 1
                run +=1
                if run > longest_run:
                    longest_run = run
                match_inds.append(i)
            else:
                if run > 0:
                    run = 0

        scores.append(score)



        if score > 5:
            logger.debug("score {}".format(score))
            logger.debug("match inds: {}".format(match_inds))
            logger.debug(first_sliced)
            logger.debug(second_sliced)

    logger.debug("longest run is {}".format(longest_run))
    logger.debug(scores)
    # Assume that most papers have titles longer than 4 words
    if max(scores) >= 4 and longest_run >= 4:
        logger.debug("max score is geq 4")
        return True

    return False

def process_similar(all_refs, similar):
    groups = []
    processed = []
    for ind, l in enumerate(similar):
        if ind in processed:
            continue
        logger.debug("--------------------------------------------------")
        logger.debug(all_refs[ind][0])
        group = []
        to_process = [ind]
        while to_process:
            logger.debug("group: {}".format(group))
            logger.debug("to process {}".format(to_process))

            check_ind = to_process.pop(0)
            group.append(check_ind)
            new_group = similar[check_ind]
            logger.debug("new group: {}".format(new_group))
            increase = len(new_group)/len(group)
            tp = set(to_process)
            ng = set(new_group)

            overlap = ng.intersection(tp)
            diff = ng.difference(tp)
            logger.debug("overlapping elements: {}, new elements: {}".format(len(overlap), len(diff)))
            logger.debug("new additions:")
            for i in diff:
                logger.debug("{}".format(all_refs[i][0]))

            # group.extend(new_group)
            for elem in new_group:
                logger.debug("checking elem {}".format(elem))
                if elem not in to_process and elem not in group:
                    logger.debug("adding this element to processing array")
                    to_process.append(elem)
                    pass
                    #logger.debug("element {} not in group, adding to to_process".format(elem))
                else:
                    pass
                    #logger.debug("element {} was already processed, or in the list to be processed.".format(elem))
        logger.debug("final group size {}".format(len(group)))

        groups.append(group)
        processed.extend(group)

    logger.info("&&&&&&&&&&&&&&&&&&&& Final groups &&&&&&&&&&&&&&&&&&&&")
    groups = sorted(groups, key=lambda x: len(x))
    for group in groups:
        logger.info("--------------------------------------------------")
        logger.info("group size: {}, members {}".format(len(group), group))
        for ref in group:
            logger.info(all_refs[ref][0])

    logger.info("Number of groups: {}".format(len(groups)))

    for ref in all_refs:
        logger.info(ref[1])


    # for ind, similar_arr in enumerate(similar):
    #     print("----------------------------------------")
    #     print(all_refs[ind][0])

    #     for sml in similar_arr:
    #         print("Cited in {}: {}".format(all_refs[sml][3], all_refs[sml][0]))

    #     if ind > 10:
    #         break

def merge_groups(groups):
    for group in groups:
        pass

def punctuation_density_separate(ref, sep_thresh=20):
    punctuation_inds = [ind for ind, char in enumerate(ref) if char in ',.']
    sep_inds = []
    sep_inds_reverse = []

    # Going forwards (with sep_inds) gives the punctuation furthest along the
    # string which conforms to the separation threshold. Use this to track the
    # start of the sequence and add it to the reverse indices, so that it gives
    # the start of the sequence
    rev_ind = None
    for ind, p_ind in enumerate(punctuation_inds):
        # At the end of the array just insert the currently tracked indices
        if ind == len(punctuation_inds) - 1:
            sep_inds.append(p_ind)
            sep_inds_reverse.append(rev_ind if rev_ind else p_ind)
            break

        # If the punctuation distance is within the threshold, then p_ind is the
        # start of a sequence
        if abs(punctuation_inds[ind+1] - p_ind) < sep_thresh:
            if not rev_ind:
                rev_ind = p_ind
            continue
        else:
            # End of the sequence
            # +1 so we don't include the actual punctuation
            sep_inds.append(p_ind + 1)
            sep_inds_reverse.append(rev_ind if rev_ind else p_ind)

            rev_ind = None

    print("--------------------------------------------------")
    print(ref)
    print("forwards: {}".format(sep_inds))
    print("reverse: {}".format(sep_inds_reverse))
    for ind, sep_ind in enumerate(sep_inds_reverse):
        if ind == 0:
            sep_str = ref[:sep_ind]
        else:
            # +1 so that we don't include the punctuation
            sep_str = ref[sep_inds[ind-1]:sep_ind+1]

        logger.debug("punctuated section {}: {}".format(ind, sep_str))

    if sep_inds:
        logger.debug(ref[sep_inds[0]:sep_inds_reverse[-1]])

    return sep_inds, sep_inds_reverse

def main():
    parser = argparse.ArgumentParser(description="Get information about references from a set of papers")
    parser.add_argument("--debug", action="store_true", help="run with debug logging on")
    parser.add_argument("--refresh", "-f", action="store_true", help="refresh the processing rather than using saved data from pickle files")
    parser.add_argument("files", nargs="+", type=str, help="text files containing text of papers")

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    f_regex = FilenameRegex("(.*) - (.*) - (.*)", author_ind=1, year_ind=2, title_ind=3)
    document_references = refgroup_from_filelist(args.files, f_regex)

    if args.refresh or not os.path.isfile("all_refs.pickle") or not os.path.isfile("similar.pickle"):
        #all_refs, similar = similar_references(document_references)
        all_refs, similar = similar_refs_slide(document_references)
        with open("all_refs.pickle", 'w') as f:
            pickle.dump(all_refs, f)
        with open("similar.pickle", 'w') as f:
            pickle.dump(similar, f)
    else:
        with open("all_refs.pickle", 'r') as f:
            all_refs = pickle.load(f)
        with open("similar.pickle", 'r') as f:
            similar = pickle.load(f)

    process_similar(all_refs, similar)

if __name__ == '__main__':
    main()
