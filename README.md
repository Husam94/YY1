# YY1 Case Study

This repository contains the code for the YY1 case study presented in the [Binning-methods](https://www.biorxiv.org/content/10.1101/2025.06.26.661884v2) and [Oyster](https://www.biorxiv.org/content/10.1101/2025.11.01.686014v1) papers


- 0.hg38.ipynb: processing the human genome version hg38
- 1.DataDownload.py: downloading the relevant .bam tracks

- R3.A.InitialProcessing.ipynb: processing the .bam files to tracks 
- R3.B.SiteSelection.ipynb: determining the sites to use for ML and preparing ML-ready data including binning and data splitting
- R3.C.Opt.ipynb: random search of oyster configurations 
- R3.D.Xps.ipynb: the experiments comparing the different model fitting methods
- R3_xps_functions.py: functions for conducting the experiments 
- R3.D.Xps.Tables.py: script for generating tables for publishing
- R3.E.Refine.ipynb: determining the contribution of different sequences and experiments with reduced combinations of sequences
- R3.F.Interp.ipynb: analysis of the final refined models, determining the contributions of DNA and histone k-mers  