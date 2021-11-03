# Predicting physiological aging rates from broad quantitative traits using machine learning

## Citation:

**Sun ED, Qian Y, Oppong R, Butler TJ, Zhao J, Chen BH, Tanaka T, Kang J, Sidore C, Cucca F, Bandinelli S, Abecasis GR, Gorospe M, et al. Predicting physiological aging rates from a range of quantitative traits using machine learning. Aging (Albany NY). 2021; 13:23471-23516. https://doi.org/10.18632/aging.203660**


This repository contains the code associated with the paper "Predicting physiological aging rates from broad quantitative traits using machine learning" by LAST NAME, et al. The code is organized into several main Python scripts:
- "AgeRatesTurnkey.py" which includes the main functions associated with processing the data, training machine learning models, and calculating physiological aging rates.
- "runModel.py" which runs the predictive framework with parameters specified in the "run_spec.txt" file.

The code is written in Python3 and requires the Numpy, Pandas, Scipy, Scikit-Learn, and Matplotlib packages among other common Python libraries. Please refer to the aforementioned scripts for a full list of the imported libraries.

In addition to the scripts, this repository contains several Jupyter notebooks which contain various code blocks and functions for analyzing the results of the predictive framework and for generating all of the key figures in the manuscript and the supplementary material. These code notebooks also contain several tests and clarifying examples.
- "results_and_analyses.ipynb" is the main notebook with functions and plots for comparing predicted to actual age, visualizing correlations between PARs, and miscellaneous controls and tests.
- "top_features.ipynb" has automatic implementations and visualizations for the three trait/feature ranking methods (added value, correlation to physiological age, and differential value between top and bottom quartiles).
- "model_comparison.ipynb" corresponds to the model comparison bar plot figure (Fig. 2a)
- "dna_methylation_age.ipynb" contains code snippets for running correlation visualizations between epigenetic aging rate (EAR) and PAR; corresponds to Fig. 5
- "reproducibility.ipynb" contains functions for visualizing correlations and reproducibility of PAR measurements across follow-up studies; corresponds to Fig. 6 and additional supplementary figures.
- "mortality.ipynb" contains code for running statistical analysis on deceased participants; corresponds to Figure 7
- "heritability.ipynb" contains code for plotting a grouped barplot of heritability values; corresponds to supplementary figure
- "gender_analysis.ipynb" contains code for boxplot of PAR by gender/sex; corresponds to supplementary figures
- "binning.ipynb" has code for plotting age distributions for different binning strategies; corresponds to supplementary figures.
- "participant_characteristics.ipynb" has code for computing basic statistics on the cohorts
- "KDM_Biological_Age.ipynb" contains code for computing the Klemera-Doubal biological age and corrected biological age
- "variability_analysis.ipynb" contains code for computing sensitivity and variability analyses for the predicted ages and feature selection procedure


NOTE: Due to sensitive information in the data, we have not included any of the required data to run the code. Interested parties should request the data directly from the SardiNIA and InCHIANTI projects:
- SardiNIA: https://sardinia.nia.nih.gov/
- InCHIANTI: http://inchiantistudy.net/wp/

Please refer to the "Code Tutorial.docx" file for a brief outline of how to run the predictive framework.