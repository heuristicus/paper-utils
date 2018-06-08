#!/usr/bin/env python
import sys
import os
import logging
import pickle
from extract_references import ReferenceGroup, refgroup_from_filelist, FilenameRegex
import pygraphviz as pgv
from graph_tool.all import *

stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'convolutional', 'neural', 'networks', 'semantic', 'segmentation']
punctuation = "-,.():;'"

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
                    
    #graph_tool
    pos = sfdp_layout(graph)
    graph_draw(graph, pos=pos, output="sfdp.pdf")

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

def similar_references(docs):
    conf_strings = ["proceedings", "conference", "journal", "transactions", "letters", "advances", "arxiv", " conf.", " proc.", "ieee", "international", "int."]
    # If the intersection of the two sets is greater than this proportion of the
    # length of the shorter set, the two references are considered similar
    overlap_prop = 0.5
    min_set_size = 5
    
    all_refs = []
    for d in docs:
        for r in d.references:
            stripped = r.lower().strip()
            conf_inds = []
            for conf_str in conf_strings:
                found_ind = stripped.find(conf_str)
                if found_ind > 0:
                    conf_inds.append(found_ind)

            if not conf_inds:
                # print("found no conference strings")
                conf_strip = stripped
            else:
                str_start = min(conf_inds)
                last_dot = stripped.rfind(".", 0, str_start)
                last_comma = stripped.rfind(",", 0, str_start)

                # print("conf string started at {}, last dot at {}, last comma at {}".format(str_start, last_dot, last_comma))
                # print("all after latest punctuation: {}".format(stripped[max(last_dot, last_comma):]))
                conf_strip = stripped[:max(last_dot, last_comma)]

            
            ref_trimmed = conf_strip.lower().translate(None, punctuation)
            ref_stopped = [word for word in ref_trimmed.split(" ") if word not in stopwords and len(word) > 2]
            ref_set = set(ref_stopped)
            all_refs.append((r, ref_stopped, ref_set, d.short_name()))


    # List for each reference of references which are similar
    similar = [[] for i in range(0, len(all_refs))]
    for cur_ind, current in enumerate(all_refs):
        print("{}/{}".format(cur_ind, len(all_refs)))
        print("--------------------------------------------------")

        cur_setlen = len(current[2])
        for other_ind, other in enumerate(all_refs):
            other_setlen = len(other[2])

            if cur_ind >= other_ind:
                continue

            if cur_setlen < min_set_size or other_setlen < min_set_size:
                continue

            max_count = min(cur_setlen, other_setlen)
            common = current[2].intersection(other[2])
            overlap = len(common)/float(max_count)
            if overlap > overlap_prop:
                similar[cur_ind].append(other_ind)
                similar[other_ind].append(cur_ind)
                # print("Overlap {} exceeds overlap proportion {}".format(overlap, overlap_prop))
                # print("{}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n{}".format(current[0], other[0]))
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                # print("{}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n{}".format(current[1], other[1]))
                # print("==================================================")

    return all_refs, similar

def process_similar(all_refs, similar):
    groups = []
    processed = []
    for ind, l in enumerate(similar):
        if ind in processed:
            continue

        group = []
        to_process = [ind]
        while to_process:
            # print("group: {}".format(group))
            # print("to process {}".format(to_process))

            check_ind = to_process.pop(0)
            group.append(check_ind)
            new_group = similar[check_ind]
#            print("checking new group at index {}: {}".format(check_ind, new_group))
            for elem in new_group:
                if elem not in to_process and elem not in group:
                    to_process.append(elem)
                    #print("element {} not in group, adding to to_process".format(elem))
                else:
                    pass
                    #print("element {} was already processed, or in the list to be processed.".format(elem))

        groups.append(group)
        processed.extend(group)

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
    
    process_similar(all_refs, similar)

if __name__ == '__main__':
    main()
    
