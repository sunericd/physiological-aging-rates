# coding: utf-8

import matplotlib as pl
pl.use('Agg')
import warnings
import os
from shutil import copyfile, copy
import AgeRatesTurnkey
from AgeRatesTurnkey import dataconf
import numpy as np
from AgeRatesTurnkey import *
import csv

# Reading from input file
with open('run_spec.txt', 'rt') as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    file_list = list(tsvin)
spec_list = file_list[1]

# Set output file directory

#outDir = input("Enter the directory where you would like to store results files: ")
outDir = spec_list[0]

if not os.path.exists(outDir):
    try:
        os.makedirs(outDir)
    except:
        print("Error opening or creating directory at " + outDir)
    if not outDir.endswith('/'):
        outDir = outDir + '/'
dataconf.results_dir = outDir
dataconf.graph_dir = outDir

# Save run_spec file for reference
copy('run_spec.txt', outDir)

# Set data file path directory
print("Enter the directory where your data is stored, NOT INCLUDING the data file itself.")
#dataDir = input("For example, if your file is in /Users/johndoe/data.tsv, then input /Users/johndoe\n")
dataDir = spec_list[1]
if not os.path.exists(dataDir):
    print("Could not find that directory.")
    sys.exit(1)
if not dataDir.endswith('/'):
    dataDir = dataDir + '/'
dataconf.SASdistRoot = dataDir

# Read in data file

#dataFile = input("Enter the full name of the data file, WITH the .tsv extension.\n")
dataFile = spec_list[2]
while not (dataFile.endswith(".tsv")):
    dataFile = input(".tsv file not recognized. Please enter a file ending with the .tsv extension.\n")
dataFile2 = dataFile
dataFile = dataFile[:-4]

fullpathsrc = dataconf.SASdistRoot+dataFile2
if os.path.isfile(fullpathsrc):
    tempFile = dataFile + "-copy.tsv"
    fullpathdest = dataconf.SASdistRoot + tempFile
    try: 
        copyfile(fullpathsrc, fullpathdest)
    except:
        print("Error manipulating file " + dataFile2)
        sys.exit(1)
    dataconf.dataFile = tempFile
else:
    print("Couldn't find file " + fullpathsrc)
    sys.exit(1)


# SET BINS

line = ""
wave = 1
bin_years = 0
while (line != "quit"):
    line = line.strip()
    #line = input("Please enter binning info, or type 'help' for help: ")
    line = spec_list[3]
    print(line)
    if (line == "quit"):
        break
    if (line == "help"):
        print("Binning info for each wave should be entered as 'bin_yrs, age_start, age_end, n_train, n_test.'")
        print("Example: '7, 24, 87, 36, 2' would signify the following.")
        print("Bin size: 7")
        print("Youngest age in sample size: 24")
        print("Oldest age in sample size: 87")
        print("# of training samples: 36")
        print("# of testing samples: 2")
#    if (line == "dev"):
#        dataconf.BestBins[1] = {6:(26,86,25,2)}
    else:
        try:
            splitArr = line.split(",")
            bin_years = int(splitArr[0])
            age_start = int(splitArr[1])
            age_end = int(splitArr[2])
            n_train = int(splitArr[3])
            n_test = int(splitArr[4])
            dataconf.BestBins[1] = {bin_years:(age_start, age_end, n_train, n_test)}
        except:
            print("Input not recognized. Please try again: ")
            continue
    if (line != "help"):
        line = "quit"

# numsplits
#splits = input("Enter number of test/train splits: ")
splits = spec_list[4]
flag = False
while (flag == False):
    try:
        numsplits = int(splits)
        flag = True
    except:
        splits = input("Number not recognized. Please try again: ")
        
    
model_name = spec_list[5]
if model_name not in dataconf.ModelInfoByModelName:
    print ("Model does not exist!")

feature_type = spec_list[6]

wavenum = 1



# Establish bin info
(age_start, age_end, n_train, n_test, bin_lbl) = dataconf.BestBinInfo (wavenum,bin_years)
print("age start: " + str(age_start) + " age end: " + str(age_end) + " n_train: " + str(n_train) + " n_test: " + str(n_test) + " bin_lbl: " + str(bin_lbl))
bin_lbl = dataconf.BinLabel (bin_years, age_start, n_train)


# In[6]:

def Graph_NTrain (title, train_range, avg_score, lda_avg_score, feature_type):
    
    # pl variables may need to be adjusted based on data
    pl.figure()
    plot_score, = pl.plot(train_range, avg_score, '#86AC41', linewidth=2.0, label="Without LDA")
    plot_lda_score, = pl.plot(train_range, lda_avg_score, '#7DA3A1', linewidth=2.0, label="With LDA (12 components)")

    best_num_feat = train_range[np.nanargmax (avg_score)]
    best_num_feat_lda = train_range[np.nanargmax (lda_avg_score)]

    max_coords = '('+str(best_num_feat)+', ' + str("%.4f" % (np.nanmax(avg_score)))+')'
    text_x = train_range[0]+(0.22*(train_range[-1]-train_range[0]))
    pl.text(text_x,0.52, 'No LDA (# individuals, max $R^2$) = '+max_coords)
    lda_max_coords = '('+str(best_num_feat_lda)+', '+str("%.4f" % (np.nanmax(lda_avg_score)))+')'
    pl.text(text_x,0.56, 'LDA (# individuals, max $R^2$) = '+lda_max_coords)
    pl.title(title)
    pl.xlabel('Number of Individuals', fontsize=14)
    pl.ylabel('Coefficient of Determination '+'( $R^2$)', fontsize=14)
    pl.ylim([0.5, 1.0])
    pl.xlim([train_range[0],train_range[-1]])
    pl.tick_params(axis='both', which='major', labelsize=12)
    pl.tick_params(axis='both', which='minor', labelsize=12)
    #pl.xticks(train_range[1], train_range[-1]+1, 20)
    pl.legend(loc='best')
    graph_path = os.path.join (dataconf.graph_dir,title+'.png')
#    graph_path = graph_path+title+'.png'
    try:
        open(graph_path, 'w')
    except OSError:
        print("Error: could not open file at" + graph_path)
    print ("Saving figure in '{}'".format(graph_path))
    pl.tight_layout()
    pl.savefig(graph_path, format='png', dpi=800)
    #pl.show()

def test_mod_samp (model_name, num_feat, wavenum, bin_years, splits, feature_type):
    model = dataconf.ModelByModelName[model_name]
    feature_func_name = 'Fisher'
    feature_func = dataconf.FeatFuncByName[feature_func_name]

    # Get n_train and n_test for this wave
    (age_start, age_end, n_train, n_test, bin_lbl) = dataconf.BestBinInfo (wavenum,bin_years)

    # Read the wave data
    (data_matrix, age_col_idx, id_col_idx, data_class_views, class_vals) = Get_Binned_Data (
        wavenum, bin_years, model_name, feature_type, age_start, age_end)
    bin_lbl = dataconf.BinLabel (bin_years, age_start, n_train)
    print ("Bin label: {}".format(bin_lbl))

    pred_label = dataconf.GetPredLabel (model_name,feature_func_name,num_feat,None,bin_lbl,feature_type,len(splits))

    scores = {}
    lda_scores = {}
    train_range = range(2, n_train, 1)

    p = ProgressBar(len (splits) * len (train_range))

    for splitnum in splits:
        for num_train in train_range:
            p.step ()

            if not num_train in scores:
                scores[num_train] = []
            if not num_train in lda_scores:
                lda_scores[num_train] = []

            split = Split()

            split.train_test(num_train, n_test, data_class_views, class_vals, age_col_idx, id_col_idx, splitnum)
            split_norm = Split().copy (split).normalize (split.train_set, split.test_set)
            fisher_weights = np.array(Fisher(split_norm))
            split.norm_weigh_sort (fisher_weights)

            new_train, new_test = split.get_trimmed_features (num_feat)
            predictions = Run_Model (model, new_train, new_test, split)

            score, p_value = pearsonr(predictions, split.test_labels)
            scores[num_train].append (score*score)

            try:
                lda_train, lda_test = lda(new_train, new_test, split)
                lda_predictions = Run_Model (model, lda_train, lda_test, split)

                lda_score, lda_p_value = pearsonr(lda_predictions, split.test_labels)
                lda_scores[num_train].append (lda_score*lda_score)
            except np.linalg.LinAlgError:
                pass

        p.step ()

        avg_scores = []
        avg_lda_scores = []
        for num_train in train_range:
            avg_scores.append (np.nanmean(scores[num_train]))
            avg_lda_scores.append (np.nanmean(lda_scores[num_train]))

    Graph_NTrain ('Sample Saturation-'+pred_label,
        train_range, avg_scores, avg_lda_scores, feature_type)




# In[7]:

# Plots graph of num_features and scores
# Saves to user inputted directory; home/suned/Graphs/Plots/ in original InChianti project
def GraphNFeat(title, num_features, avg_score, lda_avg_score):
    raw_score = (avg_score[len(avg_score)-1])
    num_feat_lda = num_features[len(num_features)-len(lda_avg_score):]
    pl.figure()
    plot_score, = pl.plot(num_features, avg_score, '#86AC41', linewidth=2.0, label="Without LDA")
    plot_lda_score, = pl.plot(num_feat_lda, lda_avg_score, '#7DA3A1', linewidth=2.0, label="With LDA (12 components)")

    best_num_feat = num_features[np.nanargmax (avg_score)]
    best_num_feat_lda = num_feat_lda[np.nanargmax (lda_avg_score)]

    max_coords = '('+str(best_num_feat)+', ' + str("%.4f" % (np.nanmax(avg_score)))+')'
    text_x = num_features[0]+(0.22*(num_features[-1]-num_features[0]))
    pl.text(text_x,0.52, 'No LDA (# features, max $R^2$): '+max_coords)
    lda_max_coords = '('+str(best_num_feat_lda)+', '+str("%.4f" % (np.nanmax(lda_avg_score)))+')'
    pl.text(text_x,0.56, 'LDA (# features, max $R^2$): '+lda_max_coords)
    pl.title(title)
    pl.xlabel('Number of Features', fontsize=14)
    pl.ylabel('Coefficient of Determination '+'( $R^2$)', fontsize=14)
    pl.ylim([0.5, 1.0])
    pl.xlim([num_features[0],num_features[-1]])
    pl.tick_params(axis='both', which='major', labelsize=12)
    pl.tick_params(axis='both', which='minor', labelsize=12)
    #pl.xticks(num_features[1], num_features[-1]+1, 20)
    pl.legend(loc='best')
    # pl.legend([plot_score, plot_lda_score, plot_benchmark], ["Without LDA", "With LDA (12 components)", "All Features"], bbox_to_anchor=(1.05, 1), loc=2)
    graph_path = os.path.join (dataconf.graph_dir,title+'.png')
#    graph_path = graph_path+title+'.png'
    try:
        open(graph_path, 'w')
    except OSError:
        print("Error: could not open file at" + graph_path)
    print ("Saving figure in '{}'".format(graph_path))
    pl.tight_layout()
    pl.savefig(graph_path, format='png', dpi=800) 
   # pl.show()


# In[8]:

def test_mod_feat (model_name, max_features, wavenum, bin_years, splits, feature_type):
    model = dataconf.ModelByModelName[model_name]
    feature_func_name = 'Fisher'
    feature_func = dataconf.FeatFuncByName[feature_func_name]


    # Get n_train and n_test for this wave
    (age_start, age_end, n_train, n_test, bin_lbl) = dataconf.BestBinInfo (wavenum,bin_years)

    # Read the wave data
    (data_matrix, age_col_idx, id_col_idx, data_class_views, class_vals) = Get_Binned_Data (
        wavenum, bin_years, model_name, feature_type, age_start, age_end)

    bin_lbl = dataconf.BinLabel (bin_years, age_start, n_train)
    print ("Bin label: {}".format(bin_lbl))

    pred_label = dataconf.GetPredLabel (model_name,feature_func_name,None,None,bin_lbl,feature_type,len(splits))
    fname_out = os.path.join (dataconf.results_dir,'BestFeatures-'+pred_label+'.txt')
    print ("BestFeatures output file: {}".format(fname_out))

    scores = []
    lda_scores = []
    num_features = range(2, max_features, 1)

    steps = len (splits)
    p = ProgressBar(steps)

    for splitnum in splits:
        p.step()

        split = Split()
        split.train_test(n_train, n_test, data_class_views, class_vals, age_col_idx, id_col_idx, splitnum)
        split_norm = Split().copy (split).normalize (split.train_set, split.test_set)
        fisher_weights = np.array(Fisher(split_norm))
        split.norm_weigh_sort (fisher_weights)
        score, lda_score = Score_Model(model, num_features, split)
        scores.append(score)
        lda_scores.append(lda_score)

    p.step ()

    avg_score = Average_Score(scores)
    lda_avg_score = Average_Score(lda_scores)

    best_num_feat = num_features[np.nanargmax (avg_score)]
    best_score = np.nanmax(avg_score)
    best_num_feat_lda = num_features[np.nanargmax (lda_avg_score)]
    best_lda_score = np.nanmax(lda_avg_score)
    print()
    res = "Best score {:.4f} @ {} features\nBest LDA score {:.4f} @ {} features\n".format(
        best_score,best_num_feat,best_lda_score,best_num_feat_lda)
    with open(fname_out, 'w') as out_file:
        out_file.write(res)
    print (res)

    GraphNFeat ('Trait Saturation-'+pred_label,
        num_features, avg_score, lda_avg_score)
    return (best_num_feat,best_score,best_num_feat_lda,best_lda_score, fname_out, pred_label)


def Graph_Actual_Predicted (pred_label, model_name, wavenum, bin_years, splits, feature_type):
    (age_start, age_end, n_train, n_test, bin_lbl) = dataconf.BestBinInfo (wavenum,bin_years)
    # Read the wave data
    (data_matrix, age_col_idx, id_col_idx, data_class_views, class_vals) = Get_Binned_Data (
        wavenum, bin_years, model_name, feature_type, age_start, age_end)

    actual_ages = data_matrix[:,age_col_idx]
    act_participant_ids = data_matrix[:,id_col_idx]

    # Read results file
    results_file_path = os.path.join (dataconf.results_dir,pred_label+'.tsv')
    results_file = np.genfromtxt(results_file_path, delimiter='\t')
    pred_ages = results_file[:,1]
    pred_participant_ids = results_file[:,0]

    # Find overlapping participants and plot
    sorted_act_ages = []
    sorted_pred_ages = []
    sorted_ids = []

    for participant in pred_participant_ids:
        if participant in act_participant_ids:
            pidx = np.where(pred_participant_ids==participant)
            aidx = np.where(act_participant_ids==participant)
            if np.isnan(pred_ages[pidx]) or np.isnan(actual_ages[aidx]):
                continue
            else:
                sorted_act_ages.append(float(actual_ages[aidx]))
                sorted_pred_ages.append(float(pred_ages[pidx]))
                sorted_ids.append(participant)

    pl.figure()
    fit = np.polyfit(sorted_act_ages,sorted_pred_ages,1)
    fit_fn = np.poly1d(fit)
    slope, intercept = np.polyfit(sorted_act_ages, fit_fn(sorted_act_ages),1)
    pl.plot(sorted_act_ages, sorted_pred_ages, 'go', sorted_act_ages, fit_fn(sorted_act_ages), '--k', alpha=0.2)
    mae = np.mean(np.abs(np.array(sorted_act_ages)-np.array(sorted_pred_ages)))
    pearson, p = pearsonr(sorted_act_ages, sorted_pred_ages)
    pl.annotate('$R^2$ = '+str("%.4f" %(pearson*pearson)), xy=(0.65, 0.325), xycoords='axes fraction')
    pl.annotate('MAE = '+str("%.4f" %(mae)), xy=(0.65, 0.25), xycoords='axes fraction')
    pl.annotate('Slope = '+str("%.4f" %slope), xy=(0.65, 0.175), xycoords='axes fraction')
    pl.annotate('y-intercept = '+str("%.2f" %intercept), xy=(0.65, 0.1), xycoords='axes fraction')
    pl.title(pred_label)
    pl.xlabel('Actual Age')
    pl.ylabel('Predicted Age')
    pl.plot([10,80], [10,80])
    save_path = os.path.join(dataconf.results_dir,pred_label+'_actVSpred.png')
    pl.savefig(save_path, format='png', dpi=150)

    pl.figure()
    age_ratio = np.array(sorted_pred_ages)/np.array(sorted_act_ages)
    fit = np.polyfit(sorted_act_ages,age_ratio,1)
    fit_fn = np.poly1d(fit)
    slope, intercept = np.polyfit(sorted_act_ages, fit_fn(sorted_act_ages),1)
    pl.plot(sorted_act_ages, age_ratio, 'go', sorted_act_ages, fit_fn(sorted_act_ages), '--k', alpha=0.2)
    pearson, p = pearsonr(sorted_act_ages, age_ratio)
    pl.annotate('$R^2$ = '+str("%.4f" %(pearson*pearson)), xy=(0.7, 0.25), xycoords='axes fraction')
    pl.annotate('Slope = '+str("%.4f" %slope), xy=(0.7, 0.175), xycoords='axes fraction')
    pl.annotate('y-intercept = '+str("%.2f" %intercept), xy=(0.7, 0.1), xycoords='axes fraction')
    pl.title(pred_label)
    pl.xlabel('Actual Age')
    pl.ylabel('Effective Rate of Aging (eRA)')
    pl.plot([10,80], [1,1])
    save_path = os.path.join(dataconf.results_dir,pred_label+'_neg_eRA.png')
    pl.savefig(save_path, format='png', dpi=150)
	
    pl.figure()
    age_accel = np.array(sorted_pred_ages)-np.array(sorted_act_ages)
    fit = np.polyfit(sorted_act_ages,age_accel,1)
    fit_fn = np.poly1d(fit)
    slope, intercept = np.polyfit(sorted_act_ages, fit_fn(sorted_act_ages),1)
    pl.plot(sorted_act_ages, age_accel, 'go', sorted_act_ages, fit_fn(sorted_act_ages), '--k', alpha=0.2)
    pearson, p = pearsonr(sorted_act_ages, age_accel)
    pl.annotate('$R^2$ = '+str("%.4f" %(pearson*pearson)), xy=(0.7, 0.25), xycoords='axes fraction')
    pl.annotate('Slope = '+str("%.4f" %slope), xy=(0.7, 0.175), xycoords='axes fraction')
    pl.annotate('y-intercept = '+str("%.2f" %intercept), xy=(0.7, 0.1), xycoords='axes fraction')
    pl.title(pred_label)
    pl.xlabel('Actual Age')
    pl.ylabel('Physiological Age Acceleration (PAA)')
    pl.plot([10,80], [1,1])
    save_path = os.path.join(dataconf.results_dir,pred_label+'_neg_PAA.png')
    pl.savefig(save_path, format='png', dpi=150)

    final_results = np.column_stack((np.array(sorted_ids), np.array(sorted_act_ages), np.array(sorted_pred_ages), np.array(age_ratio)))
    final_results_path = os.path.join(dataconf.results_dir,pred_label)
    np.savetxt(final_results_path+'.tsv', final_results, delimiter='\t')
    print ('Saved Results tsv file with columns: [ID] [Age] [Predicted Age] [Aging Rate]')

####################################################3

#model_name = 'RandForReg'
#model_name = 'WND5'
#model_name = 'RandForClf'
# model_name = 'RandForReg'
# model_name = 'KNeighClf'
# model_name = 'KNeighReg'
#model_name='elastic_net'
#model_name = "lasso"

feature_func_name = 'Fisher'
#feature_func_name = 'mRMR'
#feature_func_name = 'Pearson'

#feature_type = 'sardinia_common_cardio'
#feature_type = 'sardinia_common'
#feature_type = 'normal'


splits = range(numsplits)
max_feat = 80


np.seterr(all='ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    (best_num_feat,best_score,best_num_feat_lda,best_lda_score, fname_out, pred_label) =  test_mod_feat (
    model_name, max_feat, wavenum, bin_years, splits, feature_type)
    dataconf.BestNFeat[pred_label] = int(best_num_feat)

wavenum = 1

splits = range (numsplits)

(age_start, age_end, n_train, n_test, bin_lbl) = dataconf.BestBinInfo (wavenum,bin_years)

print(dataconf.BestNFeat)
(num_feat, doLDA) = dataconf.GetBestNFeat (model_name,feature_func_name,bin_lbl,feature_type, numsplits)


#### Comment this in to not do LDA
#doLDA=False
####

waves = range(wavenum,wavenum+1)


np.seterr(all='ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    test_mod_samp (model_name, num_feat, wavenum, bin_years, splits, feature_type)
print("End of part 2")

print("Reading data from " + fname_out)
try:
    fh = open(fname_out, 'r')
except:
    print("Error opening file at" + fname_out)
i = 0
LDAline = ""
for line in fh:
    if i == 1:
        print(line)
        LDAline = line
        break
    i += 1
splitArr = LDAline.split(" ")
value = splitArr[5] 
key = pred_label
dataconf.BestNFeat[key] = value
print(dataconf.BestNFeat)


splits = range (10)
max_feat = 80
waves = range(wavenum,wavenum+1)


# def Create_Aging_Scores (num_splits, model, num_feat, waves = range (1,4)):


waves = range(wavenum,wavenum+1)

np.seterr(all='ignore')
print(dataconf.BestNFeat)
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	aging_dict, pred_label = Get_Aging_Scores (numsplits, model_name, waves, bin_years, feature_type)


Graph_Actual_Predicted (pred_label, model_name, wavenum, bin_years, splits, feature_type)


