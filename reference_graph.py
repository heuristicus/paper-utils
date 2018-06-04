#!/usr/bin/env python
import sys
import os
from extract_references import ReferenceGroup, refgroup_from_filelist, FilenameRegex

def match_reference_to_title(docs):
    papers = [(d.title, d.author, d.year) for d in docs]
    for p in papers:
        print(p)

def main():

    f_regex = FilenameRegex("(.*) - (.*) - (.*)", author_ind=1, year_ind=2, title_ind=3)
    document_references = refgroup_from_filelist(sys.argv[1:], f_regex)
    
    match_reference_to_title(document_references)

if __name__ == '__main__':
    main()
