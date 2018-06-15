#!/usr/bin/env python
import sys
import os
import logging
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
conf_strings = ["proceedings", "conference", "journal", "transactions", "letters", "advances", "arxiv", " conf.", " proc.", "ieee", "international", "int.", "volume", "eccv", "icra", "cvpr", "iccv", " acm "]


punctuation = ",.():;'\""

def match_reference_to_title(docs):
    logger = logging.getLogger()
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

            print(conf_inds)
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
        print("{}/{}".format(cur_ind, len(all_refs)))
        print("--------------------------------------------------")
        print(all_refs[cur_ind][0])
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
                if c_list == o_list or arrays_contain_same_reference(c_list, o_list, common):
                    similar[cur_ind].append(other_ind)
                    similar[other_ind].append(cur_ind)


    return all_refs, similar

def arrays_contain_same_reference(first, second, common):
    print("\n--------------------------------------------------")
    print("Arrays not equal:")
    print(first)
    print(second)

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

    print("inds common first: {}".format(sorted(first_inds)))
    print("inds common second: {}".format(sorted(second_inds)))

    first_min = min(first_inds)
    first_max = max(first_inds)
    second_min = min(second_inds)
    second_max = max(second_inds)

    extent_first = first[first_min:first_max + 1]
    extent_second = second[second_min:second_max + 1]
    print("common extent list first: {}".format(extent_first))
    print("common extent list second: {}".format(extent_second))

    if extent_first == extent_second:
        return True
    
    # need to maybe consider more here, because there may be repeated
    # common words in one of the references, e.g. perhaps the name of
    # the journal has one of the words in it and hasn't been stripped
    # correctly.
    return False

def process_similar(all_refs, similar):
    groups = []
    processed = []
    for ind, l in enumerate(similar):
        if ind in processed:
            continue
        print("--------------------------------------------------")
        print(all_refs[ind][0])
        group = []
        to_process = [ind]
        while to_process:
            # print("group: {}".format(group))
            # print("to process {}".format(to_process))

            check_ind = to_process.pop(0)
            group.append(check_ind)
            new_group = similar[check_ind]
            increase = len(new_group)/len(group)
            tp = set(to_process)
            ng = set(new_group)
            overlap = tp.intersection(ng)
            diff = tp.difference(ng)
            print("overlapping elements: {}, new elements: {}".format(len(overlap), len(diff)))
            print("new additions:")
            for i in diff:
                print("{}".format(all_refs[i][0]))

            group.extend(new_group)
            for elem in new_group:
                if elem not in to_process and elem not in group:
                    #to_process.append(elem)
                    pass
                    #print("element {} not in group, adding to to_process".format(elem))
                else:
                    pass
                    #print("element {} was already processed, or in the list to be processed.".format(elem))
        print("final group size {}".format(len(group)))

        groups.append(group)
        processed.extend(group)

    groups = sorted(groups, key=lambda x: len(x))
    for group in groups:
        print("--------------------------------------------------")
        print("group size: {}".format(len(group)))
        for ref in group:
            print(all_refs[ref][0])

    print("Number of groups: {}".format(len(groups)))



    # for ind, similar_arr in enumerate(similar):
    #     print("----------------------------------------")
    #     print(all_refs[ind][0])

    #     for sml in similar_arr:
    #         print("Cited in {}: {}".format(all_refs[sml][3], all_refs[sml][0]))

    #     if ind > 10:
    #         break

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

    # print("--------------------------------------------------")
    # print(ref)
    # print("forwards: {}".format(sep_inds))
    # print("reverse: {}".format(sep_inds_reverse))
    # for ind, sep_ind in enumerate(sep_inds_reverse):
    #     if ind == 0:
    #         sep_str = ref[:sep_ind]
    #     else:
    #         # +1 so that we don't include the punctuation
    #         sep_str = ref[sep_inds[ind-1]:sep_ind+1]

    if sep_inds:
        print(ref[sep_inds[0]:sep_inds_reverse[-1]])

    return sep_inds, sep_inds_reverse

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    f_regex = FilenameRegex("(.*) - (.*) - (.*)", author_ind=1, year_ind=2, title_ind=3)
    document_references = refgroup_from_filelist(sys.argv[1:], f_regex)

    if not os.path.isfile("all_refs.pickle") or not os.path.isfile("similar.pickle"):
        all_refs, similar = similar_references(document_references)
        with open("all_refs.pickle", 'w') as f:
            pickle.dump(all_refs, f)
        with open("similar.pickle", 'w') as f:
            pickle.dump(similar, f)
    else:
        with open("all_refs.pickle", 'r') as f:
            all_refs = pickle.load(f)
        with open("similar.pickle", 'r') as f:
            similar = pickle.load(f)

    with open("all_refs_fullstrip.txt", 'w') as f:
        for r in all_refs:
            f.write("{}\n".format(r[5]))

    with open("all_refs_stopped.txt", 'w') as f:
        for r in all_refs:
            f.write("{}\n".format(r[1]))


    process_similar(all_refs, similar)

if __name__ == '__main__':
    main()
