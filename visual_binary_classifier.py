import os
import pickle
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from compressed_kNN import Compressed_kNN
from extract_pretrained_features_module import extract_and_collect_pretrained_features, delete_near_duplicates


class VisualBinaryClassifier:

    def __init__(self, kNN_params=None, LogReg_params=None, LightGBM_params=None, classfier_weights=[0.3,0.4,0.3],
                 models_for_features=['CLIP_ViTL_14@336','ConvNext_XL_Imagenet21k'], nromalize_features=True,
                 models_for_duplicates=['CLIP_ViTL_14@336'], duplicates_similarity_threshold=0.99, verbose=-1):

        if kNN_params is None:
            kNN_params = {}
            kNN_params['n_cols']      = 120
            kNN_params['n_rows']      = 4000
            kNN_params['n_neighbors'] = 13
            kNN_params['whiten']      = False
            kNN_params['verbose']     = verbose

        if LogReg_params is None:
            LogReg_params = {}
            LogReg_params['C']            = 1.0
            LogReg_params['class_weight'] = [0.5, 0.5]
            LogReg_params['penalty']      = 'l2'
            LogReg_params['l1_ratio']     = None

        if LightGBM_params is None:
            LGB_params = {}
            LGB_num_trees                  = 3000
            LGB_params['num_leaves']       = 8
            LGB_params['min_data_in_leaf'] = 200
            LGB_params['learning_rate']    = 0.01
            LGB_params['colsample_bytree'] = 0.25
            LGB_params['subsample']        = 0.35
            LGB_params['subsample_freq']   = 1
            LGB_params['objective']        = 'binary'
            LGB_params['metric']           = 'auc'
            LGB_params['verbose']          = verbose
        else:
            LGB_params = {}
            LGB_num_trees                  = LightGBM_params['num_trees']
            LGB_params['num_leaves']       = LightGBM_params['num_leaves']
            LGB_params['min_data_in_leaf'] = LightGBM_params['min_data_in_leaf']
            LGB_params['colsample_bytree'] = LightGBM_params['colsample_bytree']
            LGB_params['subsample']        = LightGBM_params['subsample']
            LGB_params['subsample_freq']   = LightGBM_params['subsample_freq']
            LGB_params['learning_rate']    = LightGBM_params['learning_rate']
            LGB_params['objective']        = LightGBM_params['objective']
            LGB_params['metric']           = LightGBM_params['metric']
            LGB_params['verbose']          = LightGBM_params['verbose']

        self.kNN_params = kNN_params
        self.LogReg_params = LogReg_params
        self.LGB_params = LGB_params
        self.LGB_num_trees = LGB_num_trees
        self.classfier_weights = classfier_weights
        self.verbose = verbose
        self.models_for_features = models_for_features
        self.nromalize_features = nromalize_features
        self.models_for_duplicates = models_for_duplicates
        self.duplicates_similarity_threshold = duplicates_similarity_threshold


    def fit_from_folders(self, positive_folder, negative_folder, remove_duplicates=False,
                         perform_cross_validation=False, show_cv_plots=False, reset_classifier_weights=False):

        # remove near duplicates if requested
        if remove_duplicates:
            delete_near_duplicates(positive_folder, models_to_use=self.models_for_duplicates, similarity_threshold=self.duplicates_similarity_threshold)
            delete_near_duplicates(negative_folder, models_to_use=self.models_for_duplicates, similarity_threshold=self.duplicates_similarity_threshold)

        # extract features from all images in the provided images folders
        pretrained_features_positive, _ = extract_and_collect_pretrained_features(positive_folder, models_to_use=self.models_for_features, nromalize_features=self.nromalize_features)
        pretrained_features_negative, _ = extract_and_collect_pretrained_features(negative_folder, models_to_use=self.models_for_features, nromalize_features=self.nromalize_features)

        X = np.concatenate((pretrained_features_positive, pretrained_features_negative))
        y = np.concatenate((np.ones((pretrained_features_positive.shape[0], 1)), np.zeros((pretrained_features_negative.shape[0], 1))))

        # fit the model from features
        self.fit(X, y)

        self.n_train_samples_positive = pretrained_features_positive.shape[0]
        self.n_train_samples_negative = pretrained_features_negative.shape[0]

        # if we perform cross validation, return also the GT and prediction for all valid splits
        if perform_cross_validation:
            if X.shape[0] <= 80:
                print('n_samples (positive, negative) = (%d,%d) is too small to perform cross validation. at least 80 is required' %(self.n_train_samples_positive, self.n_train_samples_negative))
                return None, None
            else:
                y_GT, y_hat = self.perform_cross_validation(X, y, show_plots=show_cv_plots, reset_classifier_weights=reset_classifier_weights, num_valid_splits=5, num_valid_repeats=1, validation_seed=123)
                return y_GT, y_hat


    def fit(self, X, y):

        num_samples = X.shape[0]

        # if the number of samples is very small, make sure to adjust defaults
        if num_samples < 800:
            self.kNN_params['n_cols']           = min(int(0.7 * num_samples), 80)
            self.kNN_params['n_rows']           = min(int(0.6 * num_samples), 500)
            self.kNN_params['n_neighbors']      = 9
            self.LogReg_params['C']             = 0.1
            self.LGB_num_trees                  = 750
            self.LGB_params['min_data_in_leaf'] = min(int(0.25 * num_samples), 100)
            self.LGB_params['subsample']        = 0.6

        if self.verbose > 0:
            print('----------------------------------------------')
            print('training kNN classifier...')
        classifier_kNN = Compressed_kNN(**self.kNN_params)
        classifier_kNN.fit(X, y)

        if self.verbose > 0:
            print('----------------------------------------------')
            print('training Logisitic Regression classifier...')
        classifier_LR = LogisticRegression(**self.LogReg_params)
        classifier_LR.fit(X, y)

        if self.verbose > 0:
            print('----------------------------------------------')
            print('training light GBM classifier...')
        classifier_LGB = lgb.train(self.LGB_params, lgb.Dataset(X, label=y), self.LGB_num_trees)

        if self.verbose > 0:
            print('----------------------------------------------')

        self.classifier_kNN = classifier_kNN
        self.classifier_LR  = classifier_LR
        self.classifier_LGB = classifier_LGB
        self.n_features = X.shape[1]


    def predict_from_folder(self, folder_to_classify, classifier_weights=None):

        pretrained_features, image_filename_map = extract_and_collect_pretrained_features(folder_to_classify, models_to_use=self.models_for_features, nromalize_features=self.nromalize_features)
        predicted_probability = self.predict(pretrained_features, classifier_weights=classifier_weights)

        return predicted_probability, image_filename_map


    def predict(self, X, classifier_weights=None):

        assert self.n_features == X.shape[1]

        if classifier_weights is None:
            classifier_weights = self.classfier_weights

        w = np.array(classifier_weights)
        w /= w.sum()

        y_hat_kNN = self.classifier_kNN.predict(X)[:,0]
        y_hat_LR  = self.classifier_LR.predict_proba(X)[:,1]
        y_hat_LGB = self.classifier_LGB.predict(X, num_iteration=self.LGB_num_trees)

        y_hat = w[0] * y_hat_kNN  + w[1] * y_hat_LR + w[2] * y_hat_LGB

        return y_hat


    def perform_cross_validation(self, X, y, show_plots=False, reset_classifier_weights=True, num_valid_splits=5, num_valid_repeats=1, validation_seed=123):
        # this eval function is a bit old and messy, but we can muscle through it

        num_samples = X.shape[0]

        # if the number of samples is very small, make sure to adjust defaults
        if num_samples < 800:
            self.kNN_params['n_cols']           = min(int(0.7 * num_samples), 80)
            self.kNN_params['n_rows']           = min(int(0.6 * num_samples), 500)
            self.kNN_params['n_neighbors']      = 9
            self.LogReg_params['C']             = 0.1
            self.LGB_num_trees                  = 750
            self.LGB_params['min_data_in_leaf'] = min(int(0.25 * num_samples), 100)
            self.LGB_params['subsample']        = 0.6

        valid_AUC_list_dict = {}
        valid_AUC_list_dict['kNN'] = []
        valid_AUC_list_dict['LR'] = []
        valid_AUC_list_dict['LGB'] = []
        valid_AUC_list_dict['ensemble'] = []
        y_valid_GT_long_dict = {}

        if num_valid_repeats > 1:
            rskf = RepeatedStratifiedKFold(n_splits=num_valid_splits, n_repeats=num_valid_repeats, random_state=validation_seed)
        else:
            rskf = StratifiedKFold(n_splits=num_valid_splits, random_state=validation_seed, shuffle=True)

        for k, (train_inds, valid_inds) in enumerate(rskf.split(X, y)):

            if self.verbose > 0:
                print('split %d:' %(k + 1))
                print('---------------------------------------------------------------------')

            # split into (train, valid) subsets
            X_train = X[train_inds]
            y_train = y[train_inds,0]

            X_valid = X[valid_inds]
            y_valid = y[valid_inds,0]

            # fit classifiers on train subset
            classfier_kNN = Compressed_kNN(**self.kNN_params)
            classfier_kNN.fit(X_train, y_train)

            classfier_LR = LogisticRegression(**self.LogReg_params)
            classfier_LR.fit(X_train, y_train)

            classifier_LGB = lgb.train(self.LGB_params, lgb.Dataset(X_train, label=y_train), self.LGB_num_trees)

            # evaluate performace - kNN
            y_train_hat_kNN = classfier_kNN.predict(X_train)[:,0]
            y_valid_hat_kNN = classfier_kNN.predict(X_valid)[:,0]

            train_Acc_kNN = ((y_train_hat_kNN > 0.5) == y_train).mean()
            valid_Acc_kNN = ((y_valid_hat_kNN > 0.5) == y_valid).mean()

            train_RMSE_kNN = np.sqrt(((y_train_hat_kNN - y_train) ** 2).mean())
            valid_RMSE_kNN = np.sqrt(((y_valid_hat_kNN - y_valid) ** 2).mean())

            train_AUC_kNN = roc_auc_score(y_train, y_train_hat_kNN)
            valid_AUC_kNN = roc_auc_score(y_valid, y_valid_hat_kNN)

            if self.verbose > 0:
                print('kNN      Accuracy (train, valid) = (%.5f, %.5f)' %(train_Acc_kNN, valid_Acc_kNN))
                print('kNN      RMSE     (train, valid) = (%.5f, %.5f)' %(train_RMSE_kNN, valid_RMSE_kNN))
                print('kNN      AUC      (train, valid) = (%.5f, %.5f)' %(train_AUC_kNN, valid_AUC_kNN))
                print('---------------------------------------------------')

            # evaluate performace - LR
            y_train_hat_LR = classfier_LR.predict_proba(X_train)[:,1]
            y_valid_hat_LR = classfier_LR.predict_proba(X_valid)[:,1]

            train_Acc_LR = ((y_train_hat_LR > 0.5) == y_train).mean()
            valid_Acc_LR = ((y_valid_hat_LR > 0.5) == y_valid).mean()

            train_RMSE_LR = np.sqrt(((y_train_hat_LR - y_train) ** 2).mean())
            valid_RMSE_LR = np.sqrt(((y_valid_hat_LR - y_valid) ** 2).mean())

            train_AUC_LR = roc_auc_score(y_train, y_train_hat_LR)
            valid_AUC_LR = roc_auc_score(y_valid, y_valid_hat_LR)

            if self.verbose > 0:
                print('LR       Accuracy (train, valid) = (%.5f, %.5f)' %(train_Acc_LR, valid_Acc_LR))
                print('LR       RMSE     (train, valid) = (%.5f, %.5f)' %(train_RMSE_LR, valid_RMSE_LR))
                print('LR       AUC      (train, valid) = (%.5f, %.5f)' %(train_AUC_LR, valid_AUC_LR))
                print('---------------------------------------------------')

            # evaluate performace - LGB
            y_train_hat_LGB = classifier_LGB.predict(X_train, num_iteration=self.LGB_num_trees)
            y_valid_hat_LGB = classifier_LGB.predict(X_valid, num_iteration=self.LGB_num_trees)

            train_Acc_LGB = ((y_train_hat_LGB > 0.5) == y_train).mean()
            valid_Acc_LGB = ((y_valid_hat_LGB > 0.5) == y_valid).mean()

            train_RMSE_LGB = np.sqrt(((y_train_hat_LGB - y_train) ** 2).mean())
            valid_RMSE_LGB = np.sqrt(((y_valid_hat_LGB - y_valid) ** 2).mean())

            train_AUC_LGB = roc_auc_score(y_train, y_train_hat_LGB)
            valid_AUC_LGB = roc_auc_score(y_valid, y_valid_hat_LGB)

            if self.verbose > 0:
                print('LGB      Accuracy (train, valid) = (%.5f, %.5f)' %(train_Acc_LGB, valid_Acc_LGB))
                print('LGB      RMSE     (train, valid) = (%.5f, %.5f)' %(train_RMSE_LGB, valid_RMSE_LGB))
                print('LGB      AUC      (train, valid) = (%.5f, %.5f)' %(train_AUC_LGB, valid_AUC_LGB))
                print('---------------------------------------------------')

            # evaluate performace - ensemble
            w = np.array(self.classfier_weights)

            y_train_hat_ensemble = w[0] * y_train_hat_kNN + w[1] * y_train_hat_LR + w[2] * y_train_hat_LGB
            y_valid_hat_ensemble = w[0] * y_valid_hat_kNN + w[1] * y_valid_hat_LR + w[2] * y_valid_hat_LGB

            train_Acc_ensemble = ((y_train_hat_ensemble > 0.5) == y_train).mean()
            valid_Acc_ensemble = ((y_valid_hat_ensemble > 0.5) == y_valid).mean()

            train_RMSE_ensemble = np.sqrt(((y_train_hat_ensemble - y_train) ** 2).mean())
            valid_RMSE_ensemble = np.sqrt(((y_valid_hat_ensemble - y_valid) ** 2).mean())

            train_AUC_ensemble = roc_auc_score(y_train, y_train_hat_ensemble)
            valid_AUC_ensemble = roc_auc_score(y_valid, y_valid_hat_ensemble)

            if self.verbose > 0:
                print('ensemble Accuracy (train, valid) = (%.5f, %.5f)' %(train_Acc_ensemble, valid_Acc_ensemble))
                print('ensemble RMSE     (train, valid) = (%.5f, %.5f)' %(train_RMSE_ensemble, valid_RMSE_ensemble))
                print('ensemble AUC      (train, valid) = (%.5f, %.5f)' %(train_AUC_ensemble, valid_AUC_ensemble))
                print('---------------------------------------------------------------------')

            # gather things for layer
            valid_AUC_list_dict['kNN'].append(valid_AUC_kNN)
            valid_AUC_list_dict['LR'].append(valid_AUC_LR)
            valid_AUC_list_dict['LGB'].append(valid_AUC_LGB)
            valid_AUC_list_dict['ensemble'].append(valid_AUC_ensemble)

            if k == 0:
                y_valid_GT_long_dict['y_GT']           = y_valid
                y_valid_GT_long_dict['y_hat_kNN']      = y_valid_hat_kNN
                y_valid_GT_long_dict['y_hat_LR']       = y_valid_hat_LR
                y_valid_GT_long_dict['y_hat_ensemble'] = y_valid_hat_ensemble
                y_valid_GT_long_dict['y_hat_LGB']      = y_valid_hat_LGB
            else:
                y_valid_GT_long_dict['y_GT']           = np.hstack((y_valid_GT_long_dict['y_GT'], y_valid))
                y_valid_GT_long_dict['y_hat_kNN']      = np.hstack((y_valid_GT_long_dict['y_hat_kNN'], y_valid_hat_kNN))
                y_valid_GT_long_dict['y_hat_LR']       = np.hstack((y_valid_GT_long_dict['y_hat_LR'], y_valid_hat_LR))
                y_valid_GT_long_dict['y_hat_LGB']      = np.hstack((y_valid_GT_long_dict['y_hat_LGB'], y_valid_hat_LGB))
                y_valid_GT_long_dict['y_hat_ensemble'] = np.hstack((y_valid_GT_long_dict['y_hat_ensemble'], y_valid_hat_ensemble))

        # validation loop ended, display performace
        self.evaluation_dict = valid_AUC_list_dict
        if self.verbose > 0:
            print('----------------------------------------------------------------')
            for key, value in valid_AUC_list_dict.items():
                print('classifier %-9s valid AUC = %.5f (+/- %.5f)' %(key, np.array(value).mean(), np.array(value).std()))
            print('----------------------------------------------------------------')

        y_GT           = y_valid_GT_long_dict['y_GT']
        y_hat_kNN      = y_valid_GT_long_dict['y_hat_kNN']
        y_hat_LR       = y_valid_GT_long_dict['y_hat_LR']
        y_hat_LGB      = y_valid_GT_long_dict['y_hat_LGB']
        y_hat_ensemble = y_valid_GT_long_dict['y_hat_ensemble']

        # show some nice summary plots for the curious user
        if self.verbose > 0 and show_plots:
            yscale = 'linear'

            thresholds = np.linspace(0,1,101)
            kNN_TP = np.zeros_like(thresholds)
            kNN_FP = np.zeros_like(thresholds)
            kNN_FN = np.zeros_like(thresholds)
            kNN_TN = np.zeros_like(thresholds)

            LR_TP = np.zeros_like(thresholds)
            LR_FP = np.zeros_like(thresholds)
            LR_FN = np.zeros_like(thresholds)
            LR_TN = np.zeros_like(thresholds)

            LGB_TP = np.zeros_like(thresholds)
            LGB_FP = np.zeros_like(thresholds)
            LGB_FN = np.zeros_like(thresholds)
            LGB_TN = np.zeros_like(thresholds)

            ensemble_TP = np.zeros_like(thresholds)
            ensemble_FP = np.zeros_like(thresholds)
            ensemble_FN = np.zeros_like(thresholds)
            ensemble_TN = np.zeros_like(thresholds)

            for k, threshold in enumerate(thresholds):
                kNN_TP[k] = (y_hat_kNN[y_GT == 1] >  threshold).mean()
                kNN_FP[k] = (y_hat_kNN[y_GT == 0] >  threshold).mean()
                kNN_FN[k] = (y_hat_kNN[y_GT == 1] <= threshold).mean()
                kNN_TN[k] = (y_hat_kNN[y_GT == 0] <= threshold).mean()

                LR_TP[k] = (y_hat_LR[y_GT == 1] >  threshold).mean()
                LR_FP[k] = (y_hat_LR[y_GT == 0] >  threshold).mean()
                LR_FN[k] = (y_hat_LR[y_GT == 1] <= threshold).mean()
                LR_TN[k] = (y_hat_LR[y_GT == 0] <= threshold).mean()

                LGB_TP[k] = (y_hat_LGB[y_GT == 1] >  threshold).mean()
                LGB_FP[k] = (y_hat_LGB[y_GT == 0] >  threshold).mean()
                LGB_FN[k] = (y_hat_LGB[y_GT == 1] <= threshold).mean()
                LGB_TN[k] = (y_hat_LGB[y_GT == 0] <= threshold).mean()

                ensemble_TP[k] = (y_hat_ensemble[y_GT == 1] >  threshold).mean()
                ensemble_FP[k] = (y_hat_ensemble[y_GT == 0] >  threshold).mean()
                ensemble_FN[k] = (y_hat_ensemble[y_GT == 1] <= threshold).mean()
                ensemble_TN[k] = (y_hat_ensemble[y_GT == 0] <= threshold).mean()

            # calc ratio curves
            ratio_baseline = 0.001
            positives_ratio_kNN = kNN_TP / (kNN_FP + ratio_baseline)
            negatives_ratio_kNN = kNN_TN / (kNN_FN + ratio_baseline)

            positives_ratio_LR = LR_TP / (LR_FP + ratio_baseline)
            negatives_ratio_LR = LR_TN / (LR_FN + ratio_baseline)

            positives_ratio_LGB = LGB_TP / (LGB_FP + ratio_baseline)
            negatives_ratio_LGB = LGB_TN / (LGB_FN + ratio_baseline)

            positives_ratio_ensemble = ensemble_TP / (ensemble_FP + ratio_baseline)
            negatives_ratio_ensemble = ensemble_TN / (ensemble_FN + ratio_baseline)

            ratio_target = 30
            positive_target_ratio_ind_kNN      = np.min(find_peaks(-np.abs(positives_ratio_kNN - ratio_target))[0])
            positive_target_ratio_ind_LR       = np.min(find_peaks(-np.abs(positives_ratio_LR - ratio_target))[0])
            positive_target_ratio_ind_LGB      = np.min(find_peaks(-np.abs(positives_ratio_LGB - ratio_target))[0])
            positive_target_ratio_ind_ensemble = np.min(find_peaks(-np.abs(positives_ratio_ensemble - ratio_target))[0])

            negative_target_ratio_ind_kNN      = np.max(find_peaks(-np.abs(negatives_ratio_kNN - ratio_target))[0])
            negative_target_ratio_ind_LR       = np.max(find_peaks(-np.abs(negatives_ratio_LR - ratio_target))[0])
            negative_target_ratio_ind_LGB      = np.max(find_peaks(-np.abs(negatives_ratio_LGB - ratio_target))[0])
            negative_target_ratio_ind_ensemble = np.max(find_peaks(-np.abs(negatives_ratio_ensemble - ratio_target))[0])

            print('----------------------------------------------------------------')
            print('To reach "target_ratio" = %d:' %(ratio_target))
            print('kNN      thresholds are: [%.2f ,%.2f]' %(thresholds[negative_target_ratio_ind_kNN],thresholds[positive_target_ratio_ind_kNN]))
            print('LR       thresholds are: [%.2f ,%.2f]' %(thresholds[negative_target_ratio_ind_LR],thresholds[positive_target_ratio_ind_LR]))
            print('LGB      thresholds are: [%.2f ,%.2f]' %(thresholds[negative_target_ratio_ind_LGB],thresholds[positive_target_ratio_ind_LGB]))
            print('ensemble thresholds are: [%.2f ,%.2f]' %(thresholds[negative_target_ratio_ind_ensemble],thresholds[positive_target_ratio_ind_ensemble]))
            print('----------------------------------------------------------------')

            ratio_ylim_mult_factor = 3

            bins = np.linspace(0,1,100)

            plt.close('all')
            plt.figure(figsize=(34,12))
            plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,hspace=0.22,wspace=0.15)

            plt.subplot(3,4,1); plt.title('kNN valid prediction histogram', fontsize=20)
            plt.hist(y_hat_kNN[y_GT == 0], bins=bins, color='red', alpha=0.8)
            plt.hist(y_hat_kNN[y_GT == 1], bins=bins, color='green', alpha=0.8)
            plt.yscale(yscale); plt.xlim(-0.01,1.01)

            plt.subplot(3,4,5); plt.title('TP(t),TN(t),FP(t),FN(t) curves', fontsize=20)
            plt.plot(thresholds, kNN_TP, color='green', lw=3.0, label='True Positive')
            plt.plot(thresholds, kNN_TN, color='red', lw=3.0, label='True Negative')
            plt.plot(thresholds, kNN_FP, color='green', lw=1.0, label='False Positive')
            plt.plot(thresholds, kNN_FN, color='red', lw=1.0, label='False Negative')
            plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.01, 0.9), ncol=1)
            plt.xlim(-0.01,1.01);

            plt.subplot(3,4,9); plt.title('TP(t)/FP(t), TN(t)/FN(t) ratios curves', fontsize=20)
            plt.plot(thresholds, positives_ratio_kNN, color='green', lw=3.0, label='True Positive / False Positive')
            plt.plot(thresholds, negatives_ratio_kNN, color='red', lw=3.0, label='True Negative / False Negative')
            plt.xlabel('threshold', fontsize=20); plt.legend(fontsize=12, loc='upper center', ncol=1)
            plt.plot([0, 100], [ratio_target, ratio_target], color='black')
            plt.xlim(-0.01,1.01); plt.ylim([0, ratio_ylim_mult_factor * ratio_target]);

            plt.subplot(3,4,2); plt.title('Logistic Regression valid prediction histogram', fontsize=20)
            plt.hist(y_hat_LR[y_GT == 0], bins=bins, color='red', alpha=0.8)
            plt.hist(y_hat_LR[y_GT == 1], bins=bins, color='green', alpha=0.8)
            plt.yscale(yscale); plt.xlim(-0.01,1.01);

            plt.subplot(3,4,6); plt.title('TP(t),TN(t),FP(t),FN(t) curves', fontsize=20)
            plt.plot(thresholds, LR_TP, color='green', lw=3.0, label='True Positive')
            plt.plot(thresholds, LR_TN, color='red', lw=3.0, label='True Negative')
            plt.plot(thresholds, LR_FP, color='green', lw=1.0, label='False Positive')
            plt.plot(thresholds, LR_FN, color='red', lw=1.0, label='False Negative')
            plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.01, 0.9), ncol=1)
            plt.xlim(-0.01,1.01);

            plt.subplot(3,4,10); plt.title('TP(t)/FP(t), TN(t)/FN(t) ratios curves', fontsize=20)
            plt.plot(thresholds, positives_ratio_LR, color='green', lw=3.0, label='True Positive / False Positive')
            plt.plot(thresholds, negatives_ratio_LR, color='red', lw=3.0, label='True Negative / False Negative')
            plt.xlabel('threshold', fontsize=20); plt.legend(fontsize=12, loc='upper center', ncol=1)
            plt.plot([0, 100], [ratio_target, ratio_target], color='black')
            plt.xlim(-0.01,1.01); plt.ylim([0, ratio_ylim_mult_factor * ratio_target]);


            plt.subplot(3,4,3); plt.title('Light GBM valid prediction histogram', fontsize=20)
            plt.hist(y_hat_LGB[y_GT == 0], bins=bins, color='red', alpha=0.8)
            plt.hist(y_hat_LGB[y_GT == 1], bins=bins, color='green', alpha=0.8)
            plt.yscale(yscale); plt.xlim(-0.01,1.01);

            plt.subplot(3,4,7); plt.title('TP(t),TN(t),FP(t),FN(t) curves', fontsize=20)
            plt.plot(thresholds, LGB_TP, color='green', lw=3.0, label='True Positive')
            plt.plot(thresholds, LGB_TN, color='red', lw=3.0, label='True Negative')
            plt.plot(thresholds, LGB_FP, color='green', lw=1.0, label='False Positive')
            plt.plot(thresholds, LGB_FN, color='red', lw=1.0, label='False Negative')
            plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.01, 0.9), ncol=1)
            plt.xlim(-0.01,1.01);

            plt.subplot(3,4,11); plt.title('TP(t)/FP(t), TN(t)/FN(t) ratios curves', fontsize=20)
            plt.plot(thresholds, positives_ratio_LGB, color='green', lw=3.0, label='True Positive / False Positive')
            plt.plot(thresholds, negatives_ratio_LGB, color='red', lw=3.0, label='True Negative / False Negative')
            plt.xlabel('threshold', fontsize=20); plt.legend(fontsize=12, loc='upper center', ncol=1)
            plt.plot([0, 100], [ratio_target, ratio_target], color='black')
            plt.xlim(-0.01,1.01); plt.ylim([0, ratio_ylim_mult_factor * ratio_target]);

            plt.subplot(3,4,4); plt.title('Ensemble valid prediction histogram', fontsize=20)
            plt.hist(y_hat_ensemble[y_GT == 0], bins=bins, color='red', alpha=0.8)
            plt.hist(y_hat_ensemble[y_GT == 1], bins=bins, color='green', alpha=0.8)
            plt.yscale(yscale); plt.xlim(-0.01,1.01);

            plt.subplot(3,4,8); plt.title('TP(t),TN(t),FP(t),FN(t) curves', fontsize=20)
            plt.plot(thresholds, ensemble_TP, color='green', lw=3.0, label='True Positive')
            plt.plot(thresholds, ensemble_TN, color='red', lw=3.0, label='True Negative')
            plt.plot(thresholds, ensemble_FP, color='green', lw=1.0, label='False Positive')
            plt.plot(thresholds, ensemble_FN, color='red', lw=1.0, label='False Negative')
            plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.01, 0.9), ncol=1)
            plt.xlim(-0.01,1.01);

            plt.subplot(3,4,12); plt.title('TP(t)/FP(t), TN(t)/FN(t) ratios curves', fontsize=20)
            plt.plot(thresholds, positives_ratio_ensemble, color='green', lw=3.0, label='True Positive / False Positive')
            plt.plot(thresholds, negatives_ratio_ensemble, color='red', lw=3.0, label='True Negative / False Negative')
            plt.xlabel('threshold', fontsize=20); plt.legend(fontsize=12, loc='upper center', ncol=1)
            plt.plot([0, 100], [ratio_target, ratio_target], color='black')
            plt.xlim(-0.01,1.01); plt.ylim([0, ratio_ylim_mult_factor * ratio_target]);


        # empirically determine the best ensemble weights
        num_weights_to_try = 10_000

        best_score = -100
        for k in range(num_weights_to_try):
            w = 0.03 + 0.97 * np.random.rand(3) # make sure minimum weight is 3%
            w /= w.sum()

            # calc ensemble using current weights
            curr_y_hat_ensemble = w[0] * y_hat_kNN + w[1] * y_hat_LR + w[2] * y_hat_LGB

            # calc AUC and Accuracy
            curr_AUC = roc_auc_score(y_GT, curr_y_hat_ensemble)
            curr_Accuracy = ((curr_y_hat_ensemble > 0.5) == y_GT).mean()

            # calc "safe margins" of "near certainly correct classification" (useful)
            thresholds = np.linspace(0,1,101)

            # calc TP/FP/FN/TN as function of threshold
            ensemble_TP = np.zeros_like(thresholds)
            ensemble_FP = np.zeros_like(thresholds)
            ensemble_FN = np.zeros_like(thresholds)
            ensemble_TN = np.zeros_like(thresholds)
            for k, threshold in enumerate(thresholds):
                ensemble_TP[k] = (curr_y_hat_ensemble[y_GT == 1] >  threshold).mean()
                ensemble_FP[k] = (curr_y_hat_ensemble[y_GT == 0] >  threshold).mean()
                ensemble_FN[k] = (curr_y_hat_ensemble[y_GT == 1] <= threshold).mean()
                ensemble_TN[k] = (curr_y_hat_ensemble[y_GT == 0] <= threshold).mean()

            # calc ratio curves
            ratio_baseline = 0.001
            positives_ratio_ensemble = ensemble_TP / (ensemble_FP + ratio_baseline)
            negatives_ratio_ensemble = ensemble_TN / (ensemble_FN + ratio_baseline)

            ratio_target = 30
            positive_target_ratio_ind_ensemble = np.min(find_peaks(-np.abs(positives_ratio_ensemble - ratio_target))[0])
            negative_target_ratio_ind_ensemble = np.max(find_peaks(-np.abs(negatives_ratio_ensemble - ratio_target))[0])

            curr_positive_target_ratio_threshold = max(0.65, thresholds[positive_target_ratio_ind_ensemble])
            curr_negative_target_ratio_threshold = min(0.35, thresholds[negative_target_ratio_ind_ensemble])

            curr_score = curr_AUC + 0.5 * curr_Accuracy + 2 * curr_negative_target_ratio_threshold + 2 * (1.0 - curr_positive_target_ratio_threshold)

            if curr_score > best_score:
                best_score = curr_score
                best_left_thresh = curr_negative_target_ratio_threshold
                best_right_thresh = curr_positive_target_ratio_threshold
                best_AUC = curr_AUC
                best_Accuracy = curr_Accuracy

                best_w = w

        self.best_weights_left_thresh  = best_left_thresh
        self.best_weights_right_thresh = best_right_thresh
        self.best_weights_AUC          = best_AUC
        self.best_weights_Accuracy     = best_Accuracy
        self.best_weights_w            = best_w

        if self.verbose > 0:
            print('------------------------------------------------------------------')
            print('for weights = %s, curr_score = %.5f' %(str(best_w), curr_score))
            print('near perfect safety margins = [%.2f, %.2f]' %(best_left_thresh, best_right_thresh))
            print('AUC, Accuracy = [%.5f, %.5f]' %(best_AUC, best_Accuracy))
            print('------------------------------------------------------------------')

        # re-set ensemble weights according to best score on validation
        if reset_classifier_weights:
            self.classfier_weights = best_w

        return y_GT, y_hat_ensemble


    def get_classifier_name(self, name_prefix=''):

        try:
            num_features = self.n_features
            num_samples = self.n_train_samples_positive + self.n_train_samples_negative
            current_date = datetime.now().strftime('%Y-%m-%d')
            output_str = '%s_num_samples_%d_num_features_%d_%s.pickle' %(name_prefix, num_samples, num_features, current_date)
        except:
            current_date = datetime.now().strftime('%Y-%m-%d')
            output_str = '%s_%s.pickle' %(name_prefix, current_date)

        return output_str


    def save_model(self, folder_path, model_prefix='good_vs_bad_classifier'):
        filename = self.get_classifier_name(name_prefix=model_prefix)
        full_path_filename = os.path.join(folder_path, filename)

        os.makedirs(folder_path, exist_ok=True)
        with open(full_path_filename, 'wb') as file:
            pickle.dump(self, file)


def load_pretrained_model(filename):

    with open(filename, 'rb') as file:
        pretrained_model = pickle.load(file)

    return pretrained_model

