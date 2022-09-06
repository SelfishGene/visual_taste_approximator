import os
import argparse

try:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
except:
    repo_dir = '/home/visual_taste_approximator/'

os.chdir(repo_dir)

import visual_binary_classifier

import warnings
warnings.simplefilter("ignore", UserWarning)


#%% parse all input arguments

parser = argparse.ArgumentParser()

parser.add_argument("--positive_folder", type=str, default="positively_labeled",
                    help="path to folder with positively labeled images")

parser.add_argument("--negative_folder", type=str, default="negatively_labeled",
                    help="path to folder with negatively labeled images")

parser.add_argument("--models_folder", type=str, default="models_folder",
                    help="path to where to store the resulting trained models")

parser.add_argument("--model_name", type=str, default="proper_portrait_classifier",
                    help="the prefix name of the model that will be saved")

parser.add_argument("--num_features_sets", type=int, default=7, choices=[1,2,4,7],
                    help="number of pretrained features sets to use (more features takes longer but more accurate)")

args = parser.parse_args()

# positive_folder = os.path.join(repo_dir, 'positively_labeled')
# negative_folder = os.path.join(repo_dir, 'negatively_labeled')
positive_folder = args.positive_folder
negative_folder = args.negative_folder

model_folder = args.models_folder
model_prefix = args.model_name
num_features_sets = args.num_features_sets

#%% set parameters and train the classifier


# good params for >> 1,000 labeled images
# these are also the default params so no need to actually define them like we do in this script
# this is provided in case one might wish to change them and this is an example how to do so

# K Nearest Neighbors params
kNN_params = {}
kNN_params['n_cols']      = 120
kNN_params['n_rows']      = 4000
kNN_params['n_neighbors'] = 13
kNN_params['whiten']      = False
kNN_params['verbose']     = 1

# Logistic Regression params
LogReg_params = {}
LogReg_params['C'] = 0.75
LogReg_params['class_weight'] = [0.5, 0.5]
LogReg_params['penalty'] = 'l2'
LogReg_params['l1_ratio'] = None

# Light Gradient Boosted Machine params
LightGBM_params = {}
LightGBM_params['num_trees']        = 3000
LightGBM_params['num_leaves']       = 8
LightGBM_params['min_data_in_leaf'] = 200
LightGBM_params['learning_rate']    = 0.01
LightGBM_params['colsample_bytree'] = 0.25
LightGBM_params['subsample']        = 0.35
LightGBM_params['subsample_freq']   = 1
LightGBM_params['objective']        = 'binary'
LightGBM_params['metric']           = 'auc'
LightGBM_params['verbose']          = -2

# ensemble classfieir weights
classfier_weights = [0.3, 0.4, 0.3]

# visual features to use for classification
nromalize_features = True

if num_features_sets == 7:
    # best classification accuracy with full 7 feature sets (~8 minutes per 1000 images on 3080 GPU)
    models_for_features = ['DINO_ViTS_8', 'DINO_ViTB_8',
                            'CLIP_ViTL_14@336', 'CLIP_ResNet50x64', 'CLIP_ViTL_14',
                            'ConvNext_XL_Imagenet21k', 'BEiT_L_16_384']
elif num_features_sets == 4:
    # slightly reduced accuracy with 4 feature sets (~4 minutes per 1000 images on 3080 GPU)
    models_for_features = ['CLIP_ViTL_14@336', 'CLIP_ResNet50x64', 'CLIP_ViTL_14', 'ConvNext_XL_Imagenet21k']
elif num_features_sets == 2:
    # some additional small reduction in accuracy (AUC = ~0.970) with 2 feature sets (~2 minutes per 1000 images on 3080 GPU)
    models_for_features = ['CLIP_ViTL_14@336', 'ConvNext_XL_Imagenet21k']
elif num_features_sets == 1:
    # some additional reduction in accuracy with 1 feature set (~1 minutes per 1000 images on 3080 GPU)
    models_for_features = ['CLIP_ViTL_14@336']

# near duplicates params
models_for_duplicates = ['CLIP_ViTL_14@336']
duplicates_similarity_threshold = 0.99

verbose = 1

binary_classifier = visual_binary_classifier.VisualBinaryClassifier(kNN_params=kNN_params,
                                                                    LogReg_params=LogReg_params,
                                                                    LightGBM_params=LightGBM_params,
                                                                    classfier_weights=classfier_weights,
                                                                    models_for_features=models_for_features,
                                                                    nromalize_features=nromalize_features,
                                                                    models_for_duplicates=models_for_duplicates,
                                                                    duplicates_similarity_threshold=duplicates_similarity_threshold,
                                                                    verbose=verbose)


remove_duplicates = False
perform_cross_validation = True
show_cv_plots = True
reset_classifier_weights = False

binary_classifier.fit_from_folders(positive_folder, negative_folder,
                                   remove_duplicates=remove_duplicates, perform_cross_validation=perform_cross_validation,
                                   show_cv_plots=show_cv_plots, reset_classifier_weights=reset_classifier_weights)


#%% save trained classfier for later

# model_folder = os.path.join(repo_dir, 'models_folder')
# model_prefix = 'proper_portrait_classifier'
model_folder = args.models_folder
model_prefix = args.model_name
binary_classifier.save_model(model_folder, model_prefix=model_prefix)

#%%
