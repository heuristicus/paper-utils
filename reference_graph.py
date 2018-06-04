#!/usr/bin/env python
import sys
import os
from extract_references import ReferenceGroup

def main():

    document_references = []

    for fname in sys.argv[1:]:
        if not os.path.isfile(fname):
            continue
        print(fname)
        reference_group = ReferenceGroup(fname)

        document_references.append(reference_group)

    for d in document_references:
        if not d.sequences or not d.all_references:
            print("no sequence/references found for {}".format(d.name))
    print("Final")
    for d in document_references:
        print(d)
        for i, ref in enumerate(d.references):
            print("{}: {}".format(i+1, ref))
        print("--------------------------------------------------")

if __name__ == '__main__':
    main()
