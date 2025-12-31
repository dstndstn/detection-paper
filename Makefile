all: detection.pdf

# fig 1: dont-coadd.pdf
# fig 2: prob-contours-a.pdf prob-rel-a.pdf

# dot -Tpdf -o dont-coadd-flow.pdf dont-coadd-flow.dot

DONT_COADD_FIGS := dont-coadd.pdf flow1.pdf flow2.pdf

BAYES_FIGS := prob-contours-a.pdf prob-rel-a.pdf prob-contours-b.pdf prob-rel-b.pdf prob-contours-c.pdf prob-1d.pdf

CHISQ_FIGS := alpha-det.pdf chisq-decision-boundary.pdf chisq-decision-boundary-sens.pdf chisq-color.pdf chisq-det-colors.pdf

SED_FIGS := image-sources-30.pdf best-color-30.pdf best-blue.pdf best-yellow.pdf best-red.pdf strength.pdf bayes-vs-gri.pdf gri-only.pdf bayes-only.pdf

GRID1_FIGS := chisq-images.pdf chisq-pos-images.pdf sed-union-images.pdf sed-mix-images.pdf unmatched-chipos-chisq-2.pdf unmatched-sed-union-chisq-2.pdf unmatched-sed-mix-chisq-2.pdf unmatched-chisq-chipos-2.pdf unmatched-sed-union-chipos-2.pdf unmatched-sed-mix-chipos-2.pdf unmatched-chisq-sed-union-2.pdf unmatched-chipos-sed-union-2.pdf unmatched-sed-mix-sed-union-2.pdf unmatched-chisq-sed-mix-2.pdf unmatched-chipos-sed-mix-2.pdf unmatched-sed-union-sed-mix-2.pdf

GRID2_FIGS := colorcolor-chisq.pdf colorcolor-chipos.pdf colorcolor-sed-union.pdf colorcolor-sed-mix.pdf unmatched-chipos-chisq-7.pdf unmatched-sed-union-chisq-7.pdf unmatched-sed-mix-chisq-7.pdf unmatched-chisq-chipos-7.pdf unmatched-sed-union-chipos-7.pdf unmatched-sed-mix-chipos-7.pdf unmatched-chisq-sed-union-7.pdf unmatched-chipos-sed-union-7.pdf unmatched-sed-mix-sed-union-7.pdf unmatched-chisq-sed-mix-7.pdf unmatched-chipos-sed-mix-7.pdf unmatched-sed-union-sed-mix-7.pdf

GRID3_FIGS := median-chisq.pdf median-chipos.pdf median-sed-union.pdf median-sed-mix.pdf unmatched-chipos-chisq-9.pdf unmatched-sed-union-chisq-9.pdf unmatched-sed-mix-chisq-9.pdf unmatched-chisq-chipos-9.pdf unmatched-sed-union-chipos-9.pdf unmatched-sed-mix-chipos-9.pdf unmatched-chisq-sed-union-9.pdf unmatched-chipos-sed-union-9.pdf unmatched-sed-mix-sed-union-9.pdf unmatched-chisq-sed-mix-9.pdf unmatched-chipos-sed-mix-9.pdf unmatched-sed-union-sed-mix-9.pdf

GAL_FIGS := galaxies.pdf galaxies-relsn.pdf

ALL_FIGS := $(DONT_COADD_FIGS) $(BAYES_FIGS) $(CHISQ_FIGS) $(SED_FIGS) $(GRID1_FIGS) $(GRID2_FIGS) $(GRID3_FIGS) $(GAL_FIGS)

detection.pdf: detection.tex $(ALL_FIGS)
	pdflatex detection
	pdflatex detection
	pdflatex detection

figs.tgz: $(ALL_FIGS)
	tar czf $@ $(ALL_FIGS)

# detection.bib detection.bbl detection.aux apj.bst \

arxiv.tgz:
	tar czf $@ detection.tex aastex63.cls \
dont-coadd.pdf sed-matched.pdf $(BAYES_FIGS) $(CHISQ_FIGS) \
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

$(BAYES_FIGS): bayes_figure.py
	python bayes_figure.py

$(CHISQ_FIGS): chi_squared_experiment.py
	python chi_squared_experiment.py

