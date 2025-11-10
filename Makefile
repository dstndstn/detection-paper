all: detection.pdf

# fig 1: dont-coadd.pdf
# fig 2: prob-contours-a.pdf prob-rel-a.pdf

# dot -Tpdf -o dont-coadd-flow.pdf dont-coadd-flow.dot

detection.pdf: detection.tex dont-coadd.pdf prob-contours-a.pdf prob-rel-a.pdf flow1.pdf flow2.pdf
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

dont-coadd.pdf: dont-coadd.py
	python dont-coadd.py

prob-contours-a.pdf prob-rel-a.pdf: bayes_figure.py
	python bayes_figure.py

flow1.pdf: flow1.dot
	dot2tex --force --preview -o flow1.tex flow1.dot
	pdflatex flow1.tex

flow2.pdf: flow2.dot
	dot2tex --force --preview -o flow2.tex flow2.dot
	pdflatex flow2.tex
