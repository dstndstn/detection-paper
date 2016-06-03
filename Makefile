all: coadd.pdf detection.pdf

coadd.pdf: coadd.tex coadd.bib
	pdflatex coadd
	pdflatex coadd
	pdflatex coadd

detection.pdf: detection.tex
	pdflatex detection
	pdflatex detection
	pdflatex detection
