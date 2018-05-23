# paper-utils
Utilities for document similarity and reference extraction for research papers

Both utilities expect input in the form of text files. If you have a directory
of pdf files, you can convert them using `pdftotext` on linux. You should use
the `-raw` switch to make sure that text in two columns is not garbled. To
convert all pdf files in the current directory to txt, outputting with the same
filename, just with a .txt extension, use

```sh
find . -type f -name *.pdf -exec pdftotext -raw {} \;
```
