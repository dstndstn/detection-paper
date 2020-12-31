all: detection.pdf

detection.pdf: detection.tex
	pdflatex detection
	pdflatex detection
	pdflatex detection

# detection.bib detection.bbl detection.aux apj.bst \

arxiv.tgz:
	tar czf $@ detection.tex aastex63.cls \
dont-coadd.pdf sed-matched.pdf prob-contours-a.pdf \
prob-rel-a.pdf prob-contours-b.pdf prob-rel-b.pdf prob-contours-c.pdf prob-1d.pdf \
image-sources-30.pdf best-color-30.pdf best-blue.pdf best-yellow.pdf best-red.pdf \
singleband.pdf strength.pdf bayes-data-cc.pdf bayes-prior-sed.pdf bayes-prior-cc.pdf \
bayes-vs-gri.pdf bayes-only.pdf gri-only.pdf galaxies.pdf galaxies-relsn.pdf
