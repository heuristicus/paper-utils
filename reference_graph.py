#!/usr/bin/env python
import sys
import os
import logging
from extract_references import ReferenceGroup, refgroup_from_filelist, FilenameRegex
import pygraphviz as pgv
from graph_tool.all import *

stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

def match_reference_to_title(docs):
    logger = logging.getLogger()
    edges = []

    graph = Graph()
    graph.add_vertex(len(docs))
    punctuation = "-,.():;'"
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

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    f_regex = FilenameRegex("(.*) - (.*) - (.*)", author_ind=1, year_ind=2, title_ind=3)
    document_references = refgroup_from_filelist(sys.argv[1:], f_regex)
    
    match_reference_to_title(document_references)

if __name__ == '__main__':
    main()
