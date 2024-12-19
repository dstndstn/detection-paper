# detection-paper
Currently arxiv:2012.15836

# Re-running experiments

DECam data are in https://github.com/dstndstn/detection-paper-data

We grabbed a subset of the Legacy Surveys DR11 data
(TODO - write script to subset the DR11 CCDs file with isin(expnum, [expnum_list]) and isin(ccdname, [ccdname_list]).)

Generate subimages with the "get-sub.py" script.

Run the pipeline with, eg,

python run-pipe.py --survey-dir detection-paper-data/ --outdir detection-paper-data/ --stage detect --radec 36.45 -4.6 --width 4000 --height 4400 --bands g,r,i -v --threads 16


