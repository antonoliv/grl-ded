#
# Makefile for feupteses/feupthesis
#
# Version: Sat May 17 22:36:50 2008
# Written by JCL
#

## Basename for result
TARGET=meic_thesis

## programs
LATEX=latex
PDFLATEX=pdflatex
PDFNUP=pdfnup
BIBTEX=bibtex

## .tex files
TEXFILES=$(wildcard *.tex)

## BibTeX files
BIB=$(wildcard *.bib)

## paper
PAPERSIZE=a4

## prefer pdflatex for bibtex
LATEXBIB=$(PDFLATEX)

## make
latexmk:
	latexmk -pdf -bibtex -recorder -pvc $(TARGET).tex

## Extensions
EXTS=aux toc idx ind ilg log out lof lot lol bbl blg brf tdo fls nav snm fdb_latexmk vrb

## clean
clean:
	for EXT in ${EXTS}; do \
	  find `pwd` -name \*\.$${EXT} -exec rm -v \{\} \; ; done

## misc
.SUFFIXES: .tex .aux .toc .lof .lot .log .dvi .pdf .bib .bbl
