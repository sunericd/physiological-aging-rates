
# coding: utf-8

import math
import pandas as pd
import numpy as np
import scipy
import os
import warnings
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import subprocess
import random as rndm
import time

import re
import pickle
try:
  get_ipython().magic('matplotlib inline')
except:
  pass


import matplotlib.pyplot as pl


# In[37]:

class dataconf :
    
    # TODO: Fix all these directories
    graph_dir = "graphs/"
    results_dir = "results/"
    DataSaveRoot = "data/"
    SASdistRoot = ""
    dataFile = ""
    SASroots = [
        SASdistRoot+"Baseline_V6/English/4.Data/SAS_Datasets/",
        SASdistRoot+"Follow-up1_V3/4.Data/SAS_Datasets/",
        SASdistRoot+"Follow-up2_V2/4.Data/SAS_Datasets/",
        SASdistRoot+"Follow-up3_V1/4.Data/SAS_Datasets/",
    ]
    SASfiles = {
        "Assays" : [
                SASroots[0]+"Assays/labo_raw.sas7bdat",
                SASroots[1]+"Assays/labf1raw.sas7bdat",
                SASroots[2]+"Assays/labf2raw.sas7bdat",
                SASroots[3]+"Assays/labf3raw.sas7bdat",
        ],
        "Diseases" : [
                SASroots[0]+"Diseases/adju_ana.sas7bdat",
                SASroots[1]+"Diseases/adjf1ana.sas7bdat",
                SASroots[2]+"Diseases/adjf2ana.sas7bdat",
                SASroots[3]+"Diseases/adjf3ana.sas7bdat",
        ],
        "Drugs" : [
                SASroots[0]+"Drugs/fmc_ana.sas7bdat",
                SASroots[1]+"Drugs/fmcf1ana.sas7bdat",
                SASroots[2]+"Drugs/fmcf2ana.sas7bdat",
                SASroots[3]+"Drugs/fmcf3ana.sas7bdat",
        ],
        "EKG_Doppler" : [
                SASroots[0]+"EKG_ENG_Doppler/mar_raw.sas7bdat",
                SASroots[1]+"EKG_ENG_Doppler/marf1raw.sas7bdat",
                SASroots[2]+"EKG_ENG_Doppler/marf2raw.sas7bdat",
                SASroots[3]+"EKG_ENG_Doppler/marf3raw.sas7bdat",
        ],
        "Interview" : [
                SASroots[0]+"Interview/int_rawe.sas7bdat",
                SASroots[1]+"Interview/inf1rawe.sas7bdat",
                SASroots[2]+"Interview/inf2rawe.sas7bdat",
                SASroots[3]+"Interview/inf3rawe.sas7bdat",
        ],
        "Med-Exam" : [
                SASroots[0]+"Medical_Exam/cli_rawe.sas7bdat",
                SASroots[1]+"Medical_Exam/clf1rawe.sas7bdat",
                SASroots[2]+"Medical_Exam/clf2rawe.sas7bdat",
                SASroots[3]+"Medical_Exam/clf3rawe.sas7bdat",
        ],
        "Nutrients-A" : [
                SASroots[0]+"Nutrients_Intake/alim_raw.sas7bdat",
                SASroots[1]+"Nutrients_Intake/alif1raw.sas7bdat",
                SASroots[2]+"Nutrients_Intake/alif2raw.sas7bdat",
                SASroots[3]+"Nutrients_Intake/alif3raw.sas7bdat",
        ],
        "Nutrients-B" : [
                SASroots[0]+"Nutrients_Intake/epic_raw.sas7bdat",
                SASroots[1]+"Nutrients_Intake/epif1raw.sas7bdat",
                SASroots[2]+"Nutrients_Intake/epif2raw.sas7bdat",
                SASroots[3]+"Nutrients_Intake/epif3raw.sas7bdat",
        ],
        "Nutrients-C" : [
                SASroots[0]+"Nutrients_Intake/nutr_raw.sas7bdat",
                SASroots[1]+"Nutrients_Intake/nutf1raw.sas7bdat",
                SASroots[2]+"Nutrients_Intake/nutf2raw.sas7bdat",
                SASroots[3]+"Nutrients_Intake/nutf3raw.sas7bdat",
        ],
        "Phys-Exam" : [
                SASroots[0]+"Physical_Exam/per_ana.sas7bdat",
                SASroots[1]+"Physical_Exam/pef1_ana.sas7bdat",
                SASroots[2]+"Physical_Exam/pef2_ana.sas7bdat",
                SASroots[3]+"Physical_Exam/pef3_ana.sas7bdat",
        ],
        # These appear to be the raw values used for computing the above
        # "Exam-B" : [
        #         SASroots[0]+"Physical_Exam/per_rawe.sas7bdat",
        #         SASroots[1]+"Physical_Exam/pef1rawe.sas7bdat",
        #         SASroots[2]+"Physical_Exam/pef2rawe.sas7bdat",
        #         SASroots[3]+"Physical_Exam/pef3rawe.sas7bdat",
        # ],
    }
    DS_types = list(SASfiles.keys())

    # TODO: Potentially drop? Have user manually clean their data, prep for direct usage in Pt04
    # These are dropped because they contain dates or ages
    drop_col_norm = [
        "CODE98",   # Unique subject ID code
        "DATA_NAS", # Date of birth/Data di nascita
        # The waves use letter codes to distinguish similar columns:
        # X: baseline, Y: Followup 1, Z: Followup 2, Q: Followup 3
        "DATEM",    # Date of Baseline instrumental exam
        "AGEM",     # Age at Baseline instrumental exam
        "DATE",     # Date of Baseline performance evaluation
        "AGE",      # Age at Baseline performance eval (years)
        "EPCDATE",  # Date of Baseline EPIC questionnaire
        "EPCAGE",   # Age at Baseline EPIC(years)/Et(anni)
        "VDATE",    # Date of Baseline medical examination
        "VAGE",     # Age at Baseline medical exam (years)
        "V1_V5",    # Month of Baseline medical examination
        "V1_V6",    # Day of Baseline medical examination
        "V1_V7",    # Year of Baseline medical examination
        "1_V7",     # Month of Baseline interview
        "1_V8",     # Day of Baseline interview
        "1_V9",     # Year of Baseline interview
        "1_V13",    # years you have been wearing glasses
        "1_V16",    # years you have been using hearing aid
        "IDATE",    # Date of Baseline Interview
        "IAGE",     # Age at the Baseline interview (years)
        "I1_V7",    # Month of Baseline interview
        "I1_V8",    # Day of Baseline interview
        "I1_V9",    # Year of Baseline interview
        "DATEL",    # Date of Baseline urine/fasting blood
        "AGEL",     # Age at Baseline fasting blood/urine(yrs)
        "CANYR",    # Year earliest cancer dx reported BL
        "IPR1YR",   # Clinical ex yr if hypertension measured
        "IPR2YR",   # Year hypertension dx reported BL
        "EPATYR",   # Year chron liver disease reported BL
        "GASTYR",   # Year gastrointest operations BL
        "ANGYR",    # Year angina dx reported Baseline
        "MIYR",     # Year myocardial infarction dx Baseline
        "CHFYR",    # Congestive heart failure year dx BL
        "STRKYR",   # Stroke or TIA year dx Baseline
        "PARKYR",   # Parkinson's disease year dx BL
        "PADIYR",   # Clinical yr if PAD measured/ROSE PAD
        "DIB1YR",   # FBG>140 or glycosuria,clin exam Baseline
        "DIB2YR",   # blood glucose>140 or diab dx year BL
        "DB1AYR",   # FBG>=126 or glycosuria,clin exam BL
        "DB2AYR",   # blood glucose>=126 or diab dx year BL
        "BPCOYR",   # Chronic bronchitis/emphysema dx yr BL
        "ASTHYR",   # Bronchial asthma dx yr Baseline
        "FFEMYR",   # Hip fracture yr dx Baseline
        "ERNDYR",   # Year herniated disc reported Baseline
        "GONAYR",   # Year knee arthritis reported FUP1
        "ANCAYR",   # Year hip arthritis reported Baseline
        "POSMYR",   # Poss osteoporosis(men),pQCT,year
        "POSFYR",   # Poss osteoporosis(women),pQCT,year
        "RL24YR",   # Impaired renal function (via 24-hour) yr
        "RENLYR",   # Impaired renal function (via C-G) year
        "FirstVisitDate", # SARDINIA
        "SecondVisitDate", # SARDINIA
        "ThirdVisitDate", # SARDINIA
        "FourthVisitDate", # SARDINIA
        "FifthVisitDate", # SARDINIA
        "pwvDate", # SARDINIA
        "Birthdate", # SARDINIA
        "Scandate", # SARDINIA
        "Subject#", # SARDINIA
        "date_neo", # SARDINIA
        "id_sir", # SARDINIA
        "id_mad", # SARDINIA
        "Wave", # SARDINIA
        "Visit", # SARDINIA
        "Occupation", # SARDINIA
        "Education", # SARDINIA
        "MaritalStatus" # SARDINIA
    ]

    common_col_norm = [
        "id_individual", # Needed to track samples
        "Age", # target variable
        "labsRBC",   #
        "labsHB", # 
        "labsMCV",    # 
        "labsMCH",     # 
        "labsWBC",     # 
        "labsPercNE",      # 
        "labsPercLY",  # 
        "labsPercMO",   # 
        "labsPercEO",    # 
        "labsPercBA",     #
        "V1_V5",
        "labsPLT",
        "labsHBF",
        "labsHBA2",
        "labsG6PD",
        "labsGlicemia",
        "labsInsulinemia",
        "labsAzotemia",
        "labsALT",
        "labsAST",
        "labsGammaGT",
        "labsFibrinogeno",
        "labsSideremia",
        "labsTransferrina",
        "labsBilirubinad",
        "labsBilirubinat",
        "labsAcidourico",
        "labsSodiemia",
        "labsPotassiemia",
        "labsVES",
        "labsPCR",
        "labsTSH",
        "labsFt4",
        "assayAdip",
        "assayLeptin",
        "assayMCP1",
        "assayIL6",
        "labsMCHC",
        "labsHtc",
        "labsMO_COUNT",
        "labsEO_COUNT",
        "labsBA_COUNT",
        "labsLY_COUNT",
        "labsNE_COUNT",
        "labsCreatinina",
        "labsColesterolo",
        "labsHDL",
        "labsTrigliceridi",
        "exmHeight",
        "exmWeight",
        "exmWaist",
        "exmHip",
        "exmBMI",
        "exmBPsys_jbs",
        "exmBPdia_jbs"
    ]

    common_cardio_col_norm = [
        "id_individual", # Needed to track samples
        "Age", # target variable
        "labsRBC",   #
        "labsHB", # 
        "labsMCV",    # 
        "labsMCH",     # 
        "labsWBC",     # 
        "labsPercNE",      # 
        "labsPercLY",  # 
        "labsPercMO",   # 
        "labsPercEO",    # 
        "labsPercBA",     #
        "V1_V5",
        "labsPLT",
        "labsHBF",
        "labsHBA2",
        "labsG6PD",
        "labsGlicemia",
        "labsInsulinemia",
        "labsAzotemia",
        "labsALT",
        "labsAST",
        "labsGammaGT",
        "labsFibrinogeno",
        "labsSideremia",
        "labsTransferrina",
        "labsBilirubinad",
        "labsBilirubinat",
        "labsAcidourico",
        "labsSodiemia",
        "labsPotassiemia",
        "labsVES",
        "labsPCR",
        "labsTSH",
        "labsFt4",
        "assayAdip",
        "assayLeptin",
        "assayMCP1",
        "assayIL6",
        "labsMCHC",
        "labsHtc",
        "labsMO_COUNT",
        "labsEO_COUNT",
        "labsBA_COUNT",
        "labsLY_COUNT",
        "labsNE_COUNT",
        "labsCreatinina",
        "labsColesterolo",
        "labsHDL",
        "labsTrigliceridi",
        "exmHeight",
        "exmWeight",
        "exmWaist",
        "exmHip",
        "exmBMI",
        "exmBPsys_jbs",
        "exmBPdia_jbs",
        #"pwvQual",
        "pwv",
        "vasPSV",
        "vasEDV",
        "vasIP",
        "vasSDratio",
        "vasAT",
        "vasvti",
        "vasSysDiam",
        "vasDiaDiam",
        "vasIMT"
    ]

    # TODO: Have the user manually input their own bins for each wave?
    # These are emirically determined using the Pt-02-Binning.ipynb notebook
    # BestBins[wavenum][bin_yrs] = (age_start, age_end, n_train, n_test)
    BestBins = {
#         0 : {
#             7 : (24,87,36,2),
#             6 : (21,87,29,2),
#             # 6 : (22,88,30,2),
#             # 6 : (23,89,31,2),
#             # 6 : (24,90,25,2),
#             # 6 : (25,85,25,2),
#         },
#         1 : {
#             7 : (27,83,33,2),
#             6 : (26,86,25,2),
#         },
#         2 : {
#             7 : (32,88,31,2),
#             6 : (29,89,25,2),
#         },
#         3 : {
#             7 : (35,84,30,2),
#             6 : (31,85,25,2),
#         },
    }
    
    # These are empirically determined using the Pt-04-BestModels.ipynb notebook
    # N.B.: As LDA features are generally better than raw, values reported are for LDA features only
    BestNFeat = {
#         'KNNc-Fsr-b7s24t36-w0' : 55,
#         'KNNc-Fsr-b7s27t33-w1' : 38,
#         'KNNc-Fsr-b7s32t31-w2' : 47,
#         'KNNc-Fsr-b7s35t30-w3' : 28,

#         'KNNc-Fsr-b6s21t29-w0' : 54, # 0.9078
#         'KNNc-Fsr-b6s26t25-w1' : 32, # 0.8317
#         'KNNc-Fsr-b6s29t25-w2' : 42, # 0.8181
#         'KNNc-Fsr-b6s31t25-w3' : 34, # 0.7693

#         'KNNc-Fsr-b6s22t30-w0' : 64, # 0.9059
#         'KNNc-Fsr-b6s23t31-w0' : 53, # 0.9065
#         'KNNc-Fsr-b6s24t25-w0' : 52, # 0.8931
#         'KNNc-Fsr-b6s25t25-w0' : 51, # 0.8847

#         'KNNr-Fsr-b6s21t29-w0' : 68, # 0.9097
#         'KNNr-Fsr-b6s26t25-w1' : 38, # 0.8359
#         'KNNr-Fsr-b6s29t25-w2' : 42, # 0.8206
#         'KNNr-Fsr-b6s31t25-w3' : 33, # 0.7714

#         'KNNr-Fsr-b6s22t30-w0' : 65, # 0.9074
#         'KNNr-Fsr-b6s23t31-w0' : 75, # 0.9096
#         'KNNr-Fsr-b6s24t25-w0' : 66, # 0.8947
#         'KNNr-Fsr-b6s25t25-w0' : 58, # 0.8873

#         'WNDc-Fsr-b6s25t25-w0' : 54,
#         'WNDc-Fsr-b6s26t25-w1' : 38, # 0.8111
#         'WNDc-Fsr-b6s29t25-w2' : 42, # 0.7888
#         'WNDc-Fsr-b6s31t25-w3' : 50, # 0.7263
        
#         'RFr-Fsr-b6s21t29-w0'  : 46, # 0.9011
#         'RFr-Fsr-b6s26t25-w1'  : 35, # 0.8325
#         'RFr-Fsr-b6s29t25-w2'  : 42, # 0.8137
#         'RFr-Fsr-b6s31t25-w3'  : 33, # 0.7643

#         'RFr-FsrNL-b6s21t29-w0'  : 58, # 0.9153 No LDA
#         'RFr-FsrNL-b6s26t25-w1'  : 56, # 0.8731 No LDA
#         'RFr-FsrNL-b6s29t25-w2'  : 75, # 0.8652 No LDA
#         'RFr-FsrNL-b6s31t25-w3'  : 66, # 0.8246 No LDA

#         'RFr-Fsr-b6s25t25-w0'   : 45, # 0.8760
#         'RFr-FsrNL-b6s25t25-w0' : 63, # 0.8988 No LDA
        
#         'RFc-Fsr-b6s21t29-w0'  : 54, # 0.8919
#         'RFc-Fsr-b6s26t25-w1'  : 36, # 0.8300
#         'RFc-Fsr-b6s29t25-w2'  : 42, # 0.8069
#         'RFc-Fsr-b6s31t25-w3'  : 33, # 0.7559

#         'RFc-FsrNL-b6s21t29-w0'  : 51, # 0.9082 No LDA
#         'RFc-FsrNL-b6s26t25-w1'  : 62, # 0.8831 No LDA
#         'RFc-FsrNL-b6s29t25-w2'  : 58, # 0.8709 No LDA
#         'RFc-FsrNL-b6s31t25-w3'  : 55, # 0.8354 No LDA
    }

    # Dicts for keeping track of feature functions, classifiers and regressors defined below
    FeatFuncInfoByName = {}
    FeatFuncByName = {}
    FeatFuncLabelByName = {}

    ModelByModelName = {}
    ModelNameByModel = {}
    ModelTypeByModel = {}
    ModelLabelByModelName = {}
 
    def NormColNames (colname):
        colRE = "^([PVIFA]?)([XYZQ])(_?)(.+)$"
        colAGERE = "(^.+)?AGE(.+$)?"
        waveletters = ["X","Y","Z","Q"]
        normcol = colname
        m = re.search(colRE, colname)
        if m and m.groups()[1] in waveletters:
            normcol = m.groups()[3]
            if re.search (colAGERE, normcol):
                normcol = 'AGE'
                
        #     print ("{} -> {}".format(colname,normcol))
        # elif m:
        #     print ("{} -> {}".format(colname,m.groups()))
        # else:
        #     print ("{}: NO RE MATCH".format(colname))
        elif colname == 'CODE98':
            normcol = 'subj_id'
        return (normcol)
    
    def SAScols(sas7bdatObj):
        h = sas7bdatObj.header.parent
        cols = []
        enc = h.encoding
        enc_err = h.encoding_errors
        for col in h.columns:
            name = col.name.decode(enc,enc_err)
            desc = col.label.decode(enc,enc_err)
            norm = dataconf.NormColNames (name)
            if col.type == 'number' and col.format == 'DATE':
                fmt = 'date'
            else:
                fmt = col.type
            cols.append ({'name':name,'desc':desc,'norm':norm, 'fmt':fmt})
        return (cols)
    
    def BinLabel (bin_years,age_start,n_train):
        return ('b{}s{}t{}'.format (bin_years,age_start,n_train))

    def BestBinInfo (wavenum,bin_years):
        age_start, age_end, n_train, n_test, bin_lbl = (None,None,None,None,None)
#         if wavenum not in [0,1,2,3]:
#             raise ValueError('wavenum {} is invalid'.format(wavenum))
#         if bin_years not in [7,6]:
#             raise ValueError('bin_years {} is invalid'.format(bin_years))

        (age_start, age_end, n_train, n_test) = dataconf.BestBins[wavenum][bin_years]
        bin_lbl = dataconf.BinLabel (bin_years,age_start,n_train)        
        return (age_start, age_end, n_train, n_test, bin_lbl)

    def GetPredLabel (model_name,feat_func_name,nfeats,doLDA,bin_lbl,feature_type,nsplits):
        print("n splits: " + str(nsplits))
        model_label = dataconf.ModelLabelByModelName[model_name]
        feat_func_label = dataconf.FeatFuncLabelByName[feat_func_name]
        if not doLDA and not doLDA is None:
            feat_func_label += 'NL'

        if (nfeats):
            feat_label = feat_func_label+str(nfeats)
        else:
            feat_label = feat_func_label

        if bin_lbl is None:
            bin_lbl = ''
        else:
            bin_lbl = '-{}'.format(bin_lbl)

        if feature_type is None:
            feat_lbl = ''
        else:
            feat_lbl = '-{}'.format(feature_type)
        if nsplits is None:
            splits_label = ''
        elif nsplits >= 1000:
            splits_label = '-s'+str(int(nsplits/1000))+'k'
        else:
            splits_label = '-s'+str(nsplits)
        pred_label = '{}-{}{}{}{}'.format(model_label,feat_label,bin_lbl,feat_lbl,splits_label)
        return (pred_label)

  
    def GetBestNFeat (model_name,feat_func_name,bin_lbl,feature_type,numsplits):
        # Why no split #?
        best_feat_key = dataconf.GetPredLabel (model_name,feat_func_name,None,True,bin_lbl,feature_type,numsplits)
        best_feat_keyNL = dataconf.GetPredLabel (model_name,feat_func_name,None,False,bin_lbl,feature_type,numsplits)
        print(best_feat_key)
        if best_feat_keyNL in dataconf.BestNFeat:
            return (dataconf.BestNFeat[best_feat_keyNL], False)
        else:
            return (dataconf.BestNFeat[best_feat_key], True)


def read_common_features(file_name):
    print ("Reading common clinical features...")
    x = pd.read_csv(file_name,sep='\t', na_values=[])
    x = x.dropna(axis=1, how='all')
    x = x.dropna(how='all')

    for feature_name in x.columns:
        if feature_name not in dataconf.common_col_norm:
            x = x.drop(feature_name, axis=1)

    threshold_num = 0.05
    for i in range (47):
        x = x.dropna(thresh=(x.shape[1]*threshold_num))
        x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
        threshold_num += 0.02
    x = x.dropna(how='any')

    return (x)

def read_common_cardio_features(file_name):
    print ("Reading common clinical features...")
    x = pd.read_csv(file_name,sep='\t', na_values=[])
    x = x.dropna(axis=1, how='all')
    x = x.dropna(how='all')

    for feature_name in x.columns:
        if feature_name not in dataconf.common_cardio_col_norm:
            x = x.drop(feature_name, axis=1)

    threshold_num = 0.05
    for i in range (47):
        x = x.dropna(thresh=(x.shape[1]*threshold_num))
        x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
        threshold_num += 0.02
    x = x.dropna(how='any')

    return (x)

def read_data_updated2(file_name, what_to_prioritize):
    x = pd.read_csv(file_name,sep='\t', na_values=[])
    x = x.dropna(axis=1, how='all')
    x = x.dropna(how='all')

    threshold_num = 0.05
    if what_to_prioritize is 'features':
        for i in range (47):
            x = x.dropna(thresh=(x.shape[1]*threshold_num))
            x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
            threshold_num += 0.02
        x = x.dropna(how='any')
    elif what_to_prioritize is 'samples':
        for i in range (47):
            x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
            x = x.dropna(thresh=(x.shape[1]*threshold_num))
            threshold_num += 0.02
        x = x.dropna(axis=1, how='any')
    else:
        return("Need to specify what to prioritize: 'features' or 'samples'")
    #x.to_csv("trimmedFile", sep='\t')

    x = clean_data(x)

    return (x)


def clean_data(x):    
    print ("Dropping features...")
    for drop_feature in dataconf.drop_col_norm:
        if drop_feature in x.columns:
            print (drop_feature)
            x = x.drop(drop_feature, axis=1)
    return (x)

def convert_data(df):
    data = df.as_matrix()
    col_names = df.columns.values
    return (data, col_names, df.columns.get_loc("Age"), df.columns.get_loc("id_individual"))


# In[13]:

def bin_data(start, end, size, data_matrix, age_col):
    bins = range (start, end+size, size)
    digitized = np.digitize(age_col, bins)

    views=[]
    for i in range (1,len(bins)):
        views.append (data_matrix[digitized == i,:])
#    views.pop()
    # initialize the class_values array (center of each bin)
    class_vals = [x+(size/2.0) for x in range(start, end, size)]
#    class_vals.pop()
    return (views, class_vals)


# In[14]:

def Average_Score (scores):
    return (np.nanmean(np.vstack(scores), axis=0))


# In[15]:

class Split (object):
    def __init__ (self):
        # These are 2D numpy arrays, features in columns, samples in rows
        self.train_set = None
        self.test_set = None
        self.n_classes = None

        # classed_labels are center values for each bin
        # labels are the continuously varying ages (unbinned) 
        self.test_labels = []
        self.train_labels = []
        self.test_classed_labels = []
        self.train_classed_labels = []
        # cached dict of class values
        self.class_vals = None
        # Participant IDs
        self.train_id = []
        self.test_id = []
        self.rand_seed = None
        
        # N.B.: Anything added here must be added to copy()!

#    def copy (self,from_sp):
#        self.train = from_sp.train
#        self.test = from_sp.test
#        self.test_labels = from_sp.test_labels
#        self.test_classed_labels = from_sp.test_classed_labels
#        self.train_labels = from_sp.train_labels
#        self.train_classed_labels = from_sp.train_classed_labels

#        self.train_vstack = from_sp.train_vstack
#        self.test_vstack = from_sp.test_vstack
#        self.train_3d = from_sp.train_3d
#        self.test_3d = from_sp.test_3d

#        self.sorted_train = from_sp.sorted_train
#        self.sorted_test = from_sp.sorted_test
#        self.stand_train = from_sp.stand_train
#        self.stand_test = from_sp.stand_test
#        self.weigh_train = from_sp.weigh_train
#        self.weigh_test = from_sp.weigh_test
#        return (self)
    
    def copy (self,from_sp):
        # This will still not make actual copies of numpys burried in lists!!!
        # They will be references to the same original numpys!
        for k in from_sp.__dict__.keys():
            if (type(from_sp.__dict__[k]) == np.ndarray):
                self.__dict__[k] = from_sp.__dict__[k].copy()
            else:
                self.__dict__[k] = from_sp.__dict__[k]
        return (self)
        
    def save (self, filename, protocol=3):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol)

    def read (self, filename):
        if sys.version_info >= (3, 0):
            split = self.read_py3_pickle(filename)
        else:
            split = self.read_py2_from_py3_pickle(filename)

        # This is to read older versions of Split
        if split.test_set is None and not split.test is None:
            split.test_set = np.vstack(split.test)
        if split.train_set is None and not split.train is None:
            split.train_set = np.vstack(split.train)
        return (self)

    def read_py3_pickle (self, filename):
        rd = None
        with open(filename, 'rb') as f:
            rd = pickle.load(f)
        return (self.copy (rd))
    
    def read_py2_from_py3_pickle (self, filename):
        rd = None
        with open(filename, 'rb') as f:
            rd = load_py2_from_py3_pickle(f)
        return (self.copy (rd))

    def train_test(self, train_amount, test_amount, views, class_vals, age_col_idx, id_col_idx, rand_seed = None):
        train = []
        test = []
        train_labels = []
        train_classed_labels = []
        test_labels = []
        test_classed_labels = []
        train_id = []
        test_id = []
        index = 0
        for view in views:
            bv = class_vals[index]
            index = index + 1 

            new_view = np.insert(view, 0, bv, axis=1)
            # This insert makes age_col_idx and id_col_idx increase by 1.
            new_age_col_idx = age_col_idx+1
            new_id_col_idx = id_col_idx+1

            # rand_seed -- NEWLY ADDED SO WE CAN RUN ON CLUSTER
            rand_seed = rndm.seed(time.time())

            if rand_seed is not None:
                random = np.random.RandomState(rand_seed).permutation(new_view)
                self.rand_seed = rand_seed
            else:
                random = np.random.permutation(new_view)
                self.rand_seed = None
            #FLOOR THIS?
            train_classed_labels.append(random[:train_amount,0])
            test_classed_labels.append(random[train_amount:train_amount+test_amount:,0])
            train_labels.append(random[:train_amount, new_age_col_idx])
            test_labels.append(random[train_amount:train_amount+test_amount:, new_age_col_idx])
            train_id.append(random[:train_amount, new_id_col_idx])
            test_id.append(random[train_amount:train_amount+test_amount:,new_id_col_idx])
            
            
            
            class_data = np.delete(random, new_age_col_idx , axis=1)
            class_data = np.delete(class_data, new_id_col_idx , axis=1)
            class_data = np.delete(class_data, 0 , axis=1)

            if train_amount+test_amount > class_data.shape[0]:
                raise Exception ("Train + Test exceeds smallest bin size of " + str(class_data.shape[0]))

            train.append(class_data[:train_amount, :])
            test.append(class_data[train_amount:train_amount+test_amount, :]) 
        
        self.train_labels = np.concatenate(train_labels)
        self.train_classed_labels = [float(n) for n in np.concatenate(train_classed_labels)]
        self.test_labels = np.concatenate(test_labels)
        self.test_classed_labels = [float(n) for n in np.concatenate(test_classed_labels)]
        self.train_id = np.concatenate(train_id)
        self.test_id = np.concatenate(test_id)
        
        #FLOOR TEST
        self.train_classed_labels = np.floor(self.train_classed_labels)
        self.test_classed_labels = np.floor(self.test_classed_labels)
        

        self.train_set = np.vstack(train)
        self.test_set = np.vstack(test)
        return (self)

    def load_wave_train_test_RS (self, wavenum, splitnum):
        wi = Config.wave_info (wavenum)
        wd = Config.wave_data (wavenum)
        self.train_test(wi.num_train, wi.num_test,
            wd.data_class_views, wd.class_vals, wd.age_col_idx, wd.id_col_idx,
            splitnum)

    def get_class_vals(self):
        if self.class_vals is None:
            class_vals_dict = {}
            for class_label in self.train_classed_labels:
                class_vals_dict[class_label] = None
            self.class_vals = [float(s) for s in sorted(class_vals_dict.keys())]
            self.n_classes = len (self.class_vals)
        return (self.class_vals)

    def get_n_classes(self):
        if self.n_classes is None:
            self.get_class_vals()
        return (self.n_classes)

    def get_train_3d (self):
        if self.train_3d is None:
            self.train_3d = get_class_mat_list (self.train_set, self.train_classed_labels)
        return (self.train_3d)

    def get_test_3d (self):
        if self.test_3d is None:
            self.test_3d = get_class_mat_list (self.test_set, self.test_classed_labels)
        return (self.test_3d)

    def sort_stand_weigh(self, feature_weights):
        i = np.argsort(feature_weights)
        sorted_train = self.get_train_vstack()[:,i]
        self.sorted_train = np.fliplr(sorted_train)
        sorted_test = self.get_test_vstack()[:,i]
        self.sorted_test = np.fliplr(sorted_test)
        self.stand(self.sorted_train, self.sorted_test)
        self.weigh_train = np.multiply(self.stand_train, feature_weights)
        self.weigh_test = np.multiply(self.stand_test, feature_weights)
        return (self)

    # Note that both of these functions sort the test/train mats by weight
    # The norm_weigh_argsort() function returns the column indexes in sort order
    # Whereas the norm_weigh_sort() returns only the sorted weight vector
    def norm_weigh_sort(self, feature_weights):
        argsort_weights = self.norm_weigh_argsort (feature_weights)
        return (feature_weights[argsort_weights])

    def norm_weigh_argsort(self, feature_weights):
        self.normalize (self.train_set, self.test_set)
        self.apply_weights (feature_weights)
        argsort_weights = self.sort_by_weight (feature_weights)
        return (argsort_weights)

    def stand_sort(self, feature_weights):
        self.stand (self.train_set, self.test_set)
        argsort_weights = self.sort_by_weight (feature_weights)
        return (feature_weights[argsort_weights])

    def stand_weigh_sort(self, feature_weights):
        self.stand (self.train_set, self.test_set)
        self.apply_weights (feature_weights)
        argsort_weights = self.sort_by_weight (feature_weights)
        return (feature_weights[argsort_weights])

    def get_trimmed_features (self, num_feat):
        new_train = self.train_set[:,:num_feat]
        new_test = self.test_set[:,:num_feat]
        return (new_train, new_test)

    def stand (self, train, test):
        scaler = StandardScaler()
        self.train_set = scaler.fit_transform(train)
        self.test_set = scaler.transform(test)
        return (self)
    
    def normalize (self, train, test):
        self.train_set = train.copy() 
        mins, maxs = normalize_by_columns (self.train_set)
        self.test_set = test.copy() 
        normalize_by_columns (self.test_set, mins, maxs)
        return (self)

    def sort_by_weight (self, feature_weights):
        argsort_weights = np.argsort(feature_weights)[::-1]
        self.train_set = self.train_set[:,argsort_weights]
        self.test_set = self.test_set[:,argsort_weights]
        return (argsort_weights)

    def apply_weights (self, feature_weights):
        self.train_set = np.multiply (self.train_set, feature_weights)
        self.test_set = np.multiply (self.test_set, feature_weights)
        return (self)


# In[16]:

dataconf.FeatFuncInfoByName = {}


# In[17]:

def Fisher(split):
    """Takes a FeatureSet_Discrete as input and calculates a Fisher score for
    each feature. Returns a newly instantiated instance of FisherFeatureWeights.

    For:
    N = number of classes
    F = number of features
    It = total number of images in training set
    Ic = number of images in a given class
    """

    if split == None:
        import inspect
        form_str = 'You passed in a None as a training set to the function {0}.{1}'	
        raise ValueError( form_str.format( cls.__name__, inspect.stack()[1][3] ) )

    # we deal with NANs/INFs separately, so turn off numpy warnings about invalid floats.
    oldsettings = np.seterr(all='ignore')

    def get_train_3d (self):
        if self.train_3d is None:
            self.train_3d = get_class_mat_list (self.train_set, self.train_classed_labels)
        return (self.train_3d)

    def get_test_3d (self):
        if self.test_3d is None:
            self.test_3d = get_class_mat_list (self.test_set, self.test_classed_labels)
        return (self.test_3d)

    #class_mats = split.get_train_3d()
    class_mats = get_class_mat_list (split.train_set, split.train_classed_labels)
    # 1D matrix 1 * F
    population_means = np.mean( split.train_set, axis = 0 )
    n_classes = class_mats.shape[0]
    n_features = split.train_set.shape[1]

    # 2D matrix shape N * F
    intra_class_means = np.empty( [n_classes, n_features] )
    # 2D matrix shape N * F
    intra_class_variances = np.empty( [n_classes, n_features] )

    class_index = 0
    for class_feature_matrix in class_mats:
        intra_class_means[ class_index ] = np.mean( class_feature_matrix, axis=0 )
    # Note that by default, numpy divides by N instead of the more common N-1, hence ddof=1.
        intra_class_variances[ class_index ] = np.var( class_feature_matrix, axis=0, ddof=1 )
        class_index += 1

    # 1D matrix 1 * F
    # we deal with NANs/INFs separately, so turn off numpy warnings about invalid floats.
    # for the record, in numpy:
    # 1./0. = inf, 0./inf = 0., 1./inf = 0. inf/0. = inf, inf/inf = nan
    # 0./0. = nan, nan/0. = nan, 0/nan = nan, nan/nan = nan, nan/inf = nan, inf/nan = nan
    # We can't deal with NANs only, must also deal with pos/neg infs
    # The masked array allows for dealing with "invalid" floats, which includes nan and +/-inf
    denom = np.mean( intra_class_variances, axis = 0 )
    denom[denom == 0] = np.nan
    feature_weights_m = np.ma.masked_invalid (
            ( np.square( population_means - intra_class_means ).sum( axis = 0 ) /
        (n_classes - 1) ) / denom
        )
    # return numpy error settings to original
    np.seterr(**oldsettings)

    # the filled(0) method of the masked array sets all nan and infs to 0
    fisher_values = feature_weights_m.filled(0).tolist()

    return (fisher_values)

dataconf.FeatFuncInfoByName['Fisher'] = [Fisher,'Fsr']

### NEWLY ADDED
def Pearson(split):
    """Calculate regression parameters and correlation statistics that fully define
    a continuous classifier.

    At present the feature weights are proportional the Pearson correlation coefficient
    for each given feature."""

    from scipy import stats

    # Known issue: running stats.linregress() with np.seterr (all='raise') has caused
    # arithmetic underflow (FloatingPointError: 'underflow encountered in stdtr' )
    # I think this is something we can safely ignore in this function, and return settings
    # back to normal at the end. -CEC
    np.seterr (under='ignore')

    if split == None:
        import inspect
        form_str = 'You passed in a None as a training set to the function {0}.{1}' 
        raise ValueError( form_str.format( cls.__name__, inspect.stack()[1][3] ) )

    matrix = split.train_set
    #FIXME: maybe add some dummyproofing to constrain incoming array size

    #r_val_sum = 0
    r_val_squared_sum = 0
    #r_val_cubed_sum = 0
    
    ages = split.train_labels

    ground_truths = np.array( [float(val) for val in ages] )
    pearson_coeffs = np.zeros(matrix.shape[1])

    for feature_index in range( matrix.shape[1] ):
        slope, intercept, pearson_coeff, p_value, std_err = stats.linregress(
            ground_truths, matrix[:,feature_index]
        )

        pearson_coeffs[feature_index] = pearson_coeff
        r_val_squared_sum += pearson_coeff * pearson_coeff

# We're just returning the pearsons^2 now...
#    pearson_values = [val*val / r_val_squared_sum for val in pearson_coeffs ]
#    pearson_coeffs = (pearson_coeffs * pearson_coeffs) / r_val_squared_sum
    pearson_coeffs *= pearson_coeffs
    

    # Reset numpy
    np.seterr (all='raise')

    return (pearson_coeffs)

dataconf.FeatFuncInfoByName['Pearson'] = [Pearson,'Prsn']



def lda(train, test, split):

    lda = LinearDiscriminantAnalysis()
    lda_train = lda.fit(train, split.train_classed_labels).transform(train)
    lda_test = lda.transform(test)
    return (lda_train, lda_test)

dataconf.FeatFuncInfoByName['LDA'] = [lda,'LDA']


# In[19]:

def mrmr(split, **kwargs):
    import subprocess
    import tempfile
    import os
    import time
    
    if 'thresh' in kwargs:
        thresh=kwargs['thresh']
    else:
        thresh=0.1
    if 'sigfigs' in kwargs:
        sigfigs=kwargs['sigfigs']
    else:
        sigfigs=7

    weights = []  
    class_labels = split.train_classed_labels.reshape(len(split.train_classed_labels),1)
    names = [float(i) for i in range (0,(split.train_set).shape[1]+1)]
    data = np.append(class_labels, split.train_set, axis=1)
    data = [names, data]
    data = np.vstack(data)
    tmpfile = tempfile.NamedTemporaryFile(delete=False).name
    # only save 5 sig-figs in text file
    np.savetxt(tmpfile, data, fmt='%.{}g'.format(sigfigs), delimiter=",")

    ignore_lines = True
    cmd_list = [Config.mrmr_bin, "-i",tmpfile, "-n", str(split.train_set.shape[1]),
                                "-s", str(split.train_set.shape[0]), '-t', str(thresh)]
    cmd_str = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).stdout
#    cmd_str = subprocess.Popen([Config.mrmr_bin, "-i","/home/suned/mrmr/test_lung_s3.csv"], stdout=subprocess.PIPE).stdout
#    t0 = time.time()
    for line in cmd_str:
#        print ("time: "+str(time.time() - t0))
#        t0 = time.time()
        line = line.decode("utf-8").strip()
#        print (line)
        # Ignore everything until lines like this:
        #    *** mRMR features *** 
        # Order 	 Fea 	 Name 	 Score
        if '*** mRMR features ***' in line:
            ignore_lines = False
            continue
        if ignore_lines:
                continue
        if 'Order' in line:
                continue
        cols = line.split()
        if (len(cols) != 4):
                continue
        try:
            cols = [int(cols[0]), int(cols[1]), float(cols[3])]
        except:
            continue
        weights.append(cols)
    if len (weights) > 0:
        weights = np.vstack(weights)
        weights = weights[weights[:,1].argsort()]
        weights = weights[:,2]
        os.unlink (tmpfile)
    else:
        raise ValueError (" ".join(cmd_list)+"\mrmr returned no weights")
    return weights

dataconf.FeatFuncInfoByName['mRMR'] = [mrmr,'mRMR']

def read_mrmr(wavenum, splitnum):
    with open(mrmr_file_path, 'rb') as f:
        mrmr_weights = pickle.load(f)
    return (mrmr_weights)


# In[20]:

# Leave this at the end of the feature function declarations
dataconf.FeatFuncByName = {k:v[0] for k, v in dataconf.FeatFuncInfoByName.items()}
dataconf.FeatFuncLabelByName = {k:v[1] for k, v in dataconf.FeatFuncInfoByName.items()}


# In[21]:

# map used to get model info by name.
# Names are populated where the functions are defined.
dataconf.ModelInfoByModelName = {}
def marg_prob_to_pred_value (marg_probs, class_vals):
    weighted = np.array(marg_probs)*np.array(class_vals)
    return (np.sum(weighted))

def marg_probs_to_pred_values (marg_probs_list, class_vals):
    weighted = marg_probs_list*np.array(class_vals)
    return (np.sum(weighted, axis=1))

def generic_classifier (clf, train_classed_data, test_classed_data, split):
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_class_labels = clf.predict(test_classed_data)
    predicted_classes = [float(s) for s in predicted_class_labels]
    class_values = [float(s) for s in clf.classes_]
    predicted_values = marg_probs_to_pred_values (clf.predict_proba(test_classed_data), class_values)

    return (predicted_classes, predicted_values)


# In[22]:

def WND5(train_classed_data, test_classed_data, classnames_list, split):
    """
    Don't call this function directly, use the wrapper functions
    DiscreteBatchClassificationResult.New() (for test sets) or
    DiscreteImageClassificationResult.NewWND5() (for single images)
    Both of these functions have dummyproofing.

    If you're using this function, your training set data is not continuous
    for N images and M features:
    trainingset is list of length L of N x M numpy matrices
    testtile is a 1 x M list of feature values
    NOTE: the trainingset and test image must have the same number of features!!!
    AND: the features must be in the same order!!
    Returns an instance of the class DiscreteImageClassificationResult
    FIXME: what about tiling??
    """
    n_test_samples = test_classed_data.shape[0]
    n_train_samples = train_classed_data.shape[0]
    predicted_classes = np.zeros(n_test_samples)
    predicted_values = np.zeros(n_test_samples)
    
    epsilon = np.finfo( np.float ).eps
    testimg_idx = 0
    trainimg_idx = 0
    
    for testimg_idx in range( n_test_samples ):
        test_class_label = split.test_classed_labels[testimg_idx]

        # initialize
        class_dists = {}
        class_counts = {}

        for trainimg_idx in range( n_train_samples ):
            train_class_label = split.train_classed_labels[trainimg_idx]
            if not train_class_label in class_dists:
                class_dists [train_class_label] = 0.0
                class_counts[train_class_label] = 0.0

            dists = np.absolute (train_classed_data [trainimg_idx] - test_classed_data [testimg_idx])
            w_dist = np.sum( dists )
            if w_dist > epsilon:
                class_counts[train_class_label] += 1.0
            else:
                continue

            w_dist = np.sum( np.square( dists ) )
            # The exponent -5 is the "5" in "WND5"
            class_dists[ train_class_label ] += w_dist ** -5

        
        class_idx = 0
        class_similarities = [0]*len(class_dists)
        for class_label in classnames_list:
            if class_counts[class_label] > 0:
                class_similarities[class_idx] = class_dists[class_label] / class_counts[class_label]
            class_idx += 1

        norm_factor = sum( class_similarities )
        marg_probs = np.array( [ x / norm_factor for x in class_similarities ] )

        predicted_class_idx = marg_probs.argmax()

        predicted_classes[testimg_idx] = classnames_list[ predicted_class_idx ]
        predicted_values[testimg_idx] = marg_prob_to_pred_value (marg_probs, classnames_list)

    return (predicted_classes, predicted_values)

dataconf.ModelInfoByModelName['WND5'] = [WND5,'C','WNDc']



def rand_forest_clf(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 30)

    predicted_classes, predicted_values = generic_classifier (clf, train_classed_data, test_classed_data, split)
    return (predicted_classes, predicted_values)

dataconf.ModelInfoByModelName['RandForClf'] = [rand_forest_clf,'C','RFc']


def k_neigh_clf(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=10, weights='distance')

    predicted_classes, predicted_values = generic_classifier (clf, train_classed_data, test_classed_data, split)
    return (predicted_classes, predicted_values)

dataconf.ModelInfoByModelName['KNeighClf'] = [k_neigh_clf,'C','KNNc']


# In[25]:

def rand_forest(train_data, train_labels, test_data, test_labels):
    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor(n_estimators=30)
    forest.fit(train_data, train_labels)
    predicted = forest.predict(test_data)
    return (predicted)

dataconf.ModelInfoByModelName['RandForReg'] = [rand_forest,'R','RFr']


def k_neigh(train_data, train_labels, test_data, test_labels):
    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor(n_neighbors = 20, weights='distance', p=1)
    neigh.fit(train_data, train_labels)
    predicted = neigh.predict(test_data)
    return (predicted)
	
dataconf.ModelInfoByModelName['KNeighReg'] = [k_neigh,'R','KNNr']


def SVR(train_data, train_labels, test_data, test_labels):
    from sklearn.svm import SVR
    regr = SVR()
    regr.fit(train_data, train_labels)
    predicted = regr.predict(test_data)
    return (predicted)

dataconf.ModelInfoByModelName['SVR'] = [SVR,'R','SVR']


def elastic_net(train_data, train_labels, test_data, test_labels):
    from sklearn.linear_model import ElasticNet
    regr = ElasticNet()
    regr.fit(train_data, train_labels)
    predicted = regr.predict(test_data)
    return (predicted)

dataconf.ModelInfoByModelName['elastic_net'] = [elastic_net,'R','eln']


def lasso(train_data, train_labels, test_data, test_labels):
    from sklearn.linear_model import ElasticNet
    regr = ElasticNet(l1_ratio=1.0)
    regr.fit(train_data, train_labels)
    predicted = regr.predict(test_data)
    return (predicted)

dataconf.ModelInfoByModelName['lasso'] = [lasso,'R','lasso']


def linreg(train_data, train_labels, test_data, test_labels):
    from sklearn.linear_model import LinearRegression
    regr = LinearRegression()
    regr.fit(train_data, train_labels)
    predicted = regr.predict(test_data)
    return (predicted)

dataconf.ModelInfoByModelName['linreg'] = [linreg,'R','linreg']

def xgboost(train_data, train_labels, test_data, test_labels):
    from xgboost import XGBRegressor
    regr = XGBRegressor()
    regr.fit(train_data, train_labels)
    predicted = regr.predict(test_data)
    return (predicted)

dataconf.ModelInfoByModelName['xgboost'] = [xgboost,'R','xgboost']



dataconf.ModelByModelName = {k:v[0] for k, v in dataconf.ModelInfoByModelName.items()}
dataconf.ModelNameByModel = {v[0]:k for k, v in dataconf.ModelInfoByModelName.items()}
dataconf.ModelTypeByModel = {v[0]:v[1] for k, v in dataconf.ModelInfoByModelName.items()}
dataconf.ModelLabelByModelName = {k:v[2] for k, v in dataconf.ModelInfoByModelName.items()}


# In[28]:

def normalize_by_columns ( full_stack, mins = None, maxs = None ):
    """This is a global function to normalize a matrix by columns.
    If numpy 1D arrays of mins and maxs are provided, the matrix will be normalized against these ranges
    Otherwise, the mins and maxs will be determined from the matrix, and the matrix will be normalized
    against itself. The mins and maxs will be returned as a tuple.
    Out of range matrix values will be clipped to min and max (including +/- INF)
    zero-range columns will be set to 0.
    NANs in the columns will be set to 0.
    The normalized output range is hard-coded to 0-100
    """
    # Edge cases to deal with:
    # Range determination:
    # 1. features that are nan, inf, -inf
    # max and min determination must ignore invalid numbers
    # nan -> 0, inf -> max, -inf -> min
    # Normalization:
    # 2. feature values outside of range
    # values clipped to range (-inf to min -> min, max to inf -> max) - leaves nan as nan
    # 3. feature ranges that are 0 result in nan feature values
    # 4. all nan feature values set to 0

    # Turn off numpy warnings, since we're taking care of invalid values explicitly
    oldsettings = np.seterr(all='ignore')
    if (mins is None or maxs is None):
        # mask out NANs and +/-INFs to compute min/max
        full_stack_m = np.ma.masked_invalid (full_stack, copy=False)
        maxs = full_stack_m.max (axis=0)
        mins = full_stack_m.min (axis=0)

    # clip the values to the min-max range (NANs are left, but +/- INFs are taken care of)
    full_stack.clip (mins, maxs, full_stack)
    # remake a mask to account for NANs and divide-by-zero from max == min
    full_stack_m = np.ma.masked_invalid (full_stack, copy=False)

    # Normalize
    full_stack_m -= mins
    full_stack_m /= (maxs - mins)
    # Left over NANs and divide-by-zero from max == min become 0
    # Note the deep copy to change the numpy parameter in-place.
    full_stack[:] = full_stack_m.filled (0) * 100.0

    # return settings to original
    np.seterr(**oldsettings)

    return (mins,maxs)


# In[29]:

def get_class_mat_list (mat, class_labels):
    assert (len(mat) == len(class_labels))
    class_label_dict = {}
    class_mats = []
    class_label_idx = 0
    for samp_idx in range (len (mat)):
        class_label = class_labels[samp_idx]
        if not class_label in class_label_dict:
            class_label_dict[class_label] = class_label_idx
            class_mats.append (mat[samp_idx])
            class_label_idx += 1
        else:
            class_idx = class_label_dict[class_label]
            class_mats[class_idx] = np.vstack ([class_mats[class_idx],mat[samp_idx]])
    return (np.array(class_mats))
    


# In[34]:

def Get_Binned_Data (wavenum, bin_years, model_name, feature_type, age_start=None, age_end=None):
    in_path = dataconf.SASdistRoot+dataconf.dataFile
    print ('Reading data from {}'.format(in_path))
    if feature_type == 'sardinia_common':
        df = read_common_features(in_path)
    elif feature_type == 'sardinia_common_cardio':
        df = read_common_cardio_features(in_path)
    else:
        df = read_data_updated2 (in_path,'features')
    print ('Shape after removing all blank rows, cols, blank ages: {}'.format(df.shape))
    data_matrix, col_names, age_col_idx, id_col_idx = convert_data (df)
    age_col = data_matrix[:,age_col_idx]
    n_features = len (col_names) - 2
    print ("n_features = {}".format(n_features))
	
	# runs elastic net model from R on cleaned data if needed
    #if model_name is "elastic_net":
    #    import subprocess
    #    np.savetxt("cleanedData.csv", data_matrix, delimiter=",")
    #    subprocess.call (["./elastic_net.R", "--vanilla", "elastic_net.R", str(age_col_idx), str(id_col_idx)])
    #    sys.exit()
		
    if age_start is None or age_end is None:
        (age_start, age_end, n_train, n_test, bin_lbl) = dataconf.BestBinInfo (wavenum,bin_years)

    data_class_views, class_vals = bin_data (age_start, age_end, bin_years, data_matrix, age_col)
    return (data_matrix, age_col_idx, id_col_idx, data_class_views, class_vals)



# In[31]:

def Run_Model (model, train, test, split):
    if dataconf.ModelTypeByModel[model] == 'C':
        cls_names, predictions = model (train, test, split.get_class_vals(), split)
    else:
        predictions = model (train, split.train_labels, test, split.test_labels)
    return (predictions)

# Varies number of features, reports R^2 for model and model+LDA
# N.B.: Assumes features are already sorted by weight!
def Score_Model(model, feature_range, split):
    scores = np.array([None]*len(feature_range), dtype=np.float64)
    lda_scores = np.array([None]*len(feature_range), dtype=np.float64)
    idx = 0
    n_classes = split.get_n_classes()
    class_vals = split.get_class_vals()
    for feature_num in feature_range:
        new_train, new_test = split.get_trimmed_features (feature_num)
        predictions = Run_Model (model, new_train, new_test, split)
        score, p_value = pearsonr(predictions, split.test_labels)
        scores[idx] = score*score
        if feature_num>n_classes-2:
            try:
                lda_train, lda_test = lda(new_train, new_test, split)
                lda_predictions = Run_Model (model, lda_train, lda_test, split)
                lda_score, lda_p_value = pearsonr(lda_predictions, split.test_labels)
                lda_scores[idx] = lda_score*lda_score
            except np.linalg.LinAlgError:
                pass
        idx += 1

    return(scores, lda_scores)



# In[32]:

# Regression strategy based on "Best Aging Rates" notebook
# Works for classifier and regressor models
# When read_splits is False, uses the split number as the random seed.

def Create_Aging_Scores (num_splits, model_name, num_feat, waves, bin_years, feature_type, doLDA = True):
    model = dataconf.ModelByModelName[model_name]
    feature_func_name = 'Fisher'
    feature_func = dataconf.FeatFuncByName[feature_func_name]

    aging_scores = {}
    pearsons = {}
    # Change to range(1,4) for all 3 waves
    steps = (num_splits*len(waves))+len(waves)
    p = ProgressBar(steps)
    for wavenum in waves:

        # Get n_train and n_test for this wave
        (age_start, age_end, n_train, n_test, bin_lbl) = dataconf.BestBinInfo (wavenum,bin_years)

        # Read the wave data
        (data_matrix, age_col_idx, id_col_idx, data_class_views, class_vals) = Get_Binned_Data (
            wavenum, bin_years, model_name, feature_type, age_start, age_end)


        bin_lbl = dataconf.BinLabel (bin_years, age_start, n_train)
        print ("Bin label: {}".format(bin_lbl))

        for splitnum in range(num_splits):
            p.step ()
            split = Split()
            split.train_test(n_train, n_test, data_class_views, class_vals, age_col_idx, id_col_idx, splitnum)
            split_norm = Split().copy (split).normalize (split.train_set, split.test_set)
            fisher_weights = np.array(Fisher(split_norm))
            split.norm_weigh_sort (fisher_weights)

            new_train, new_test = split.get_trimmed_features (num_feat)

            try:
                if (doLDA):
                    lda_train, lda_test = lda(new_train, new_test, split)
                    predictions = Run_Model (model, lda_train, lda_test, split)
                else:
                    predictions = Run_Model (model, new_train, new_test, split)

                n_test_samples = len(split.test_classed_labels)
                if not wavenum in pearsons:
                    pearsons[wavenum] = []
                pearsons[wavenum].append(pearsonr(predictions, split.test_labels)[0])
                for test_idx in range(n_test_samples):
                    sample_id = split.test_id[test_idx]
                    if not sample_id in aging_scores:
                        aging_scores[sample_id] = {}
                    wave = 'W{}'.format(wavenum)
                    if not wave in aging_scores[sample_id]:
                        aging_scores[sample_id][wave] = {'A':split.test_labels[test_idx], 'P':[], 'mP':None}
                    aging_scores[sample_id][wave]['P'].append(predictions[test_idx])
            except np.linalg.LinAlgError as err:
                print ('split',split.get_fname (wavenum),err)
                sys.stdout.flush()

        p.step ()

        pred_label = dataconf.GetPredLabel (model_name,feature_func_name,num_feat,doLDA,bin_lbl,feature_type,num_splits)
        file_path = os.path.join (dataconf.results_dir,pred_label+'.tsv')
        print ("\nWriting results file '{}'".format (file_path))
        wave_smpls = []
        wave_preds = []
        wave_scores = {}
        with open(file_path, 'w') as the_file:
            for sample_id in sorted(aging_scores):
                mean = aging_scores[sample_id][wave]['mP'] = np.mean(aging_scores[sample_id][wave]['P'])
                the_file.write('{}\t{}\n'.format(sample_id,mean))
                wave_smpls.append (aging_scores[sample_id][wave]['A'])
                wave_preds.append(mean)
                wave_scores[sample_id] = dict(aging_scores[sample_id][wave])
        print('Wave {}'.format(wavenum))
        pears_means = pearsonr(wave_preds, wave_smpls)[0]
        pears_means *= pears_means
        mean_pears = np.sum(np.square(pearsons[wavenum]))/len(pearsons[wavenum])
        print('R^2 (pred means): {:.3f}'.format (pears_means))
        print('Mean R^2 (preds): {:.3f}'.format (mean_pears))

        file_path = os.path.join (dataconf.results_dir,pred_label+'.pickle')
        print ("\nWriting pickle to '{}'".format (file_path))
        
        with open(file_path, 'wb') as the_file:
            pickle.dump(wave_scores, the_file, protocol=3)

        sys.stdout.flush()

    p.step ()
    return (aging_scores, pred_label)

def Get_Aging_Scores (num_splits, model_name, waves, bin_years, feature_type):
    feature_func_name = 'Fisher'
    feature_func = dataconf.FeatFuncByName[feature_func_name]
    print("num_splits aging scores method: " + str(num_splits))
    wave_scores = {}
    for wavenum in waves:
        bin_lbl = dataconf.BestBinInfo (wavenum,bin_years)[4]
        (num_feat, doLDA) = dataconf.GetBestNFeat (model_name,feature_func_name,bin_lbl,feature_type, num_splits)
        num_feat = int(num_feat)
        pred_label = dataconf.GetPredLabel (model_name,feature_func_name,num_feat,doLDA,bin_lbl,feature_type,num_splits)
        file_path = os.path.join (dataconf.results_dir,pred_label+'.pickle')
        try:
            print ("Trying to read from {}... ".format(file_path), end="")
            with open(file_path, 'rb') as the_file:
                wave_scores[wavenum] = pickle.load(the_file)
            print ("Success.")
        except OSError:
            print ("Re-calculating...")
            wave_scores[wavenum], pred_label = Create_Aging_Scores (num_splits, model_name, num_feat, waves, bin_years, feature_type, doLDA)
            try:
                print ("Trying to read from {}... ".format(file_path), end="")
                with open(file_path, 'rb') as the_file:
                    wave_scores[wavenum] = pickle.load(the_file)
                print ("Success.")
            except OSError:
                raise ValueError('Failed - quitting')
    
    aging_scores = {}
    for wavenum in wave_scores:
        wave = 'W{}'.format(wavenum)
        wave_smpls = []
        wave_preds = []
        for sample_id in wave_scores[wavenum]:
            if sample_id not in aging_scores:
                aging_scores[sample_id] = {}
            if wave not in aging_scores[sample_id]:
                aging_scores[sample_id][wave] = {}
            aging_scores[sample_id][wave] = wave_scores[wavenum][sample_id]
            wave_smpls.append (aging_scores[sample_id][wave]['A'])
            wave_preds.append(aging_scores[sample_id][wave]['mP'])
        pears_means = pearsonr(wave_preds, wave_smpls)[0]
        print('Wave {}, R^2 (pred means): {:.3f}'.format (wavenum,pears_means*pears_means))

    return (aging_scores, pred_label)


# In[33]:

import sys, time

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class ProgressBar(object):
    def __init__(self, iterations):
        self.iterations = iterations
        self.cur_iter = 0
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 20
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_ipython

    def step(self):
        if self.cur_iter < self.iterations:
            self.animate (self.cur_iter)

    def animate_ipython(self, cur_iter):
        self.update_iteration(cur_iter + 1)
        print ('\r', self, end="")
        if (cur_iter + 1 >= self.iterations):
            print ()
        sys.stdout.flush()

    def update_iteration(self, cur_iter):
        self.__update_amount(cur_iter)
        self.prog_bar += '  %4d of %4d complete ' % (cur_iter, self.iterations)

    def __update_amount(self, new_amount):
        self.cur_iter = new_amount
        frac_done = float(new_amount) / float(self.iterations)
        all_full = self.width
        num_hashes = int(round(frac_done * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        percent_done = round(frac_done*100.0)
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] +             (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)








