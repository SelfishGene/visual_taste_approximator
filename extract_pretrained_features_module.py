import os
import glob
import time
import shutil
import pickle
import timm
import clip
import sklearn
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms as pth_transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from compressed_kNN import Compressed_kNN

import warnings
warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)

#%% helper functions


def load_timm_model(model_name='convnext_xlarge_in22k', device='cpu'):

    pretrained_model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    model_config_dict = resolve_data_config({}, model=pretrained_model)
    model_preprocess = create_transform(**model_config_dict)

    return pretrained_model, model_preprocess


def load_dino_model(model_name='dino_vitb8', device='cpu'):

    model_preprocess = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    pretrained_model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)

    return pretrained_model, model_preprocess


def extract_pretrained_features(base_image_folder, model_to_use='CLIP_ViTL_14@336'):

    # assume that the folder contains only images (in terms of file endings)
    all_image_filenames = glob.glob(os.path.join(base_image_folder, '*.*'))
    if len(all_image_filenames) == 0:
        all_image_filenames = glob.glob(os.path.join(base_image_folder, 'images', '*.*'))
        transfer_images = False
        if len(all_image_filenames) == 0:
            print('no images inside requested folder (or subfolder named "images/)')
            return
    else:
        transfer_images = True

    print('calculating %d features of model "%s"' %(len(all_image_filenames), model_to_use))

    # create subfolder structure if needed
    images_folder = os.path.join(base_image_folder, 'images')
    features_folder = os.path.join(base_image_folder, 'pretrained_features')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(features_folder, exist_ok=True)

    # transfer images to their correct folder if needed
    if transfer_images:
        for full_image_filename in all_image_filenames:
            if os.path.isfile(full_image_filename):
                shutil.copy(full_image_filename, images_folder)
                os.remove(full_image_filename)
        all_image_filenames = glob.glob(os.path.join(base_image_folder, 'images', '*.*'))

    # load requested model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if   model_to_use == 'CLIP_ViTL_14@336':
        pretrained_model, model_preprocess = clip.load("ViT-L/14@336px", device=device)
    elif model_to_use == 'CLIP_ViTL_14':
        pretrained_model, model_preprocess = clip.load("ViT-L/14", device=device)
    elif model_to_use == 'CLIP_ViTB_16':
        pretrained_model, model_preprocess = clip.load("ViT-B/16", device=device)
    elif model_to_use == 'CLIP_ViTB_32':
        pretrained_model, model_preprocess = clip.load("ViT-B/32", device=device)
    elif model_to_use == 'CLIP_ResNet50x64':
        pretrained_model, model_preprocess = clip.load("RN50x64", device=device)
    elif model_to_use == 'CLIP_ResNet50x16':
        pretrained_model, model_preprocess = clip.load("RN50x16", device=device)
    elif model_to_use == 'CLIP_ResNet50x4':
        pretrained_model, model_preprocess = clip.load("RN50x4", device=device)
    elif model_to_use == 'CLIP_ResNet50x1':
        pretrained_model, model_preprocess = clip.load("RN50", device=device)
    elif model_to_use == 'CLIP_ResNet101':
        pretrained_model, model_preprocess = clip.load("RN101", device=device)

    elif model_to_use == 'DINO_ResNet50':
        pretrained_model, model_preprocess = load_dino_model("dino_resnet50", device=device)
    elif model_to_use == 'DINO_ViTS_8':
        pretrained_model, model_preprocess = load_dino_model("dino_vits8", device=device)
    elif model_to_use == 'DINO_ViTB_8':
        pretrained_model, model_preprocess = load_dino_model("dino_vitb8", device=device)

    elif model_to_use == 'ConvNext_XL_Imagenet21k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_xlarge_in22k', device=device)
    elif model_to_use == 'ConvNext_XL_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_xlarge_384_in22ft1k', device=device)
    elif model_to_use == 'ConvNext_L_Imagenet21k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_large_in22k', device=device)
    elif model_to_use == 'ConvNext_L_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_large_384_in22ft1k', device=device)

    elif model_to_use == 'EffNet_L2_NS_475':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnet_l2_ns_475', device=device)
    elif model_to_use == 'EffNet_B7_NS_600':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnet_b7_ns', device=device)
    elif model_to_use == 'EffNetV2_L_480_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnetv2_l_in21ft1k', device=device)
    elif model_to_use == 'EffNetV2_S_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnetv2_s_in21ft1k', device=device)

    elif model_to_use == 'BEiT_L_16_512':
        pretrained_model, model_preprocess = load_timm_model(model_name='beit_large_patch16_512', device=device)
    elif model_to_use == 'BEiT_L_16_384':
        pretrained_model, model_preprocess = load_timm_model(model_name='beit_large_patch16_384', device=device)
    elif model_to_use == 'BEiT_L_16_224':
        pretrained_model, model_preprocess = load_timm_model(model_name='beit_large_patch16_224', device=device)

    elif model_to_use == 'DeiT3_L_16_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='deit3_large_patch16_384_in21ft1k', device=device)
    elif model_to_use == 'DeiT3_H_14_224_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='deit3_huge_patch14_224_in21ft1k', device=device)
    elif model_to_use == 'DeiT3_L_16_224_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='deit3_large_patch16_224_in21ft1k', device=device)
    else:
        print('unrecognized modelname, not calculated any features!')
        return


    start_time = time.time()
    # go over all images and append features to features dict
    for k, curr_image_filename in enumerate(all_image_filenames):
        curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]
        curr_features_dict_filename = os.path.join(features_folder, curr_sample_name + '.pickle')

        if (k + 1) % 1000 == 0:
            duration_min = (time.time() - start_time) / 60
            print('extracted "%s" features thus far from %d images. took %.2f minutes' %(model_to_use, k + 1, duration_min))

        # check if features_dict file exists, if it doesn't, create one
        if os.path.isfile(curr_features_dict_filename):
            curr_features_dict = pickle.load(open(curr_features_dict_filename, "rb"))
        else:
            curr_features_dict = {}

        # if the requested features were already calculated for this sample, skip it
        if model_to_use in curr_features_dict.keys():
            continue

        # extract the features
        curr_image_PIL = Image.open(curr_image_filename)
        curr_image_PIL = curr_image_PIL.convert("RGB")

        with torch.no_grad():
            if 'CLIP' in model_to_use:
                curr_pretrained_features = pretrained_model.encode_image(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'DINO' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'ConvNext' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'EffNet' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'BEiT' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'DeiT' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))

        curr_features_dict[model_to_use] = curr_pretrained_features.detach().cpu().numpy()

        # save the dictionary
        pickle.dump(curr_features_dict, open(curr_features_dict_filename, "wb"))

    return


def collect_pretrained_features(base_image_folder, requested_features_model='CLIP_ViTL_14@336', nromalize_features=True):
    # this function assumes that the folder stucture is proper and features dict contains the requested features

    images_folder = os.path.join(base_image_folder, 'images')
    features_folder = os.path.join(base_image_folder, 'pretrained_features')
    all_feature_dict_filenames = glob.glob(os.path.join(features_folder, '*.pickle'))
    all_image_filenames = glob.glob(os.path.join(images_folder, '*.*'))

    try:
        curr_features_dict = pickle.load(open(all_feature_dict_filenames[0], "rb"))
        num_features = curr_features_dict[requested_features_model].shape[1]
    except:
        print('the requested features were not calculated.')
        return [],[]

    num_images = len(all_feature_dict_filenames)

    # create matrix to fill
    pretrained_image_features_matrix = np.zeros((num_images, num_features))

    # go over all samples and collect the features
    image_filename_map = {}
    for k, curr_image_filename in enumerate(all_image_filenames):
        curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]
        curr_features_dict_filename = os.path.join(features_folder, curr_sample_name + '.pickle')
        curr_features_dict = pickle.load(open(curr_features_dict_filename, "rb"))
        pretrained_image_features_matrix[k,:] = curr_features_dict[requested_features_model]
        image_filename_map[k] = curr_image_filename

    # normalize features to unit norm
    if nromalize_features:
        pretrained_image_features_matrix /= np.linalg.norm(pretrained_image_features_matrix, axis=1, keepdims=True)

    return pretrained_image_features_matrix, image_filename_map


def extract_and_collect_pretrained_features(images_base_folder, models_to_use=['CLIP_ViTL_14@336','CLIP_ResNet50x64'], nromalize_features=True):
    # this function will extract the features of all models in "models_to_use", collect the  and concatenate them

    # extracting features
    for model_to_use in models_to_use:
        extract_pretrained_features(images_base_folder, model_to_use=model_to_use)

    # collecting features
    features_list = []
    image_filename_map_list = []
    for requested_features_model in models_to_use:
        image_features, image_filename_map = collect_pretrained_features(images_base_folder, requested_features_model=requested_features_model, nromalize_features=nromalize_features)
        features_list.append(image_features)
        image_filename_map_list.append(image_filename_map)

    # make sure the maps are identical
    try:
        for k in range(len(image_filename_map_list) - 1):
            for key in image_filename_map_list[k].keys():
                assert image_filename_map_list[k][key] == image_filename_map_list[k + 1][key]
    except:
        print('the maps are not identical. quitting')
        return

    # concatenate the features
    combined_image_features = np.concatenate(features_list, axis=1)

    return combined_image_features, image_filename_map_list[0]


def delete_worst_near_duplicates(base_image_folder, good_vs_bad_classifier, models_to_use=['CLIP_ViTL_14@336','CLIP_ResNet50x64'], similarity_threshold=0.99, minibatch_size=10_000):
    # this function will apply classifier on all images in folder, and among near duplicates will remove the ones with lowest classifier prediction
    # this function does not assume "proper" folder stucture, but will create it and calculate features if necessary

    assert good_vs_bad_classifier.models_for_features == models_to_use, 'error, this will not work if the features are not the same and in the same order'

    features_folder = os.path.join(base_image_folder, 'pretrained_features')

    # collect the requested features to calculate near duplication based on
    image_features, image_filename_map = extract_and_collect_pretrained_features(base_image_folder, models_to_use=models_to_use, nromalize_features=True)
    similarity_threshold = len(models_to_use) * similarity_threshold

    # apply classifier and sort the images from "low" to "high" so that low will be removed first
    predicted_probability = good_vs_bad_classifier.predict(image_features)
    sorting_order = np.argsort(predicted_probability)
    image_features = image_features[sorting_order]
    image_filename_map_sorted = {}
    for k in range(sorting_order.shape[0]):
        image_filename_map_sorted[k] = image_filename_map[sorting_order[k]]
        # print(k, predicted_probability[sorting_order[k]])

    image_filename_map = image_filename_map_sorted
    total_num_samples = image_features.shape[0]
    num_batches = np.ceil(total_num_samples / minibatch_size).astype(int)

    feature_inds_to_drop = []

    end_row_ind = 0
    for batch_ind in range(num_batches):
        start_row_ind = end_row_ind
        end_row_ind = min(start_row_ind + minibatch_size, total_num_samples)
        image_feature_curr_batch = image_features[start_row_ind:end_row_ind]
        curr_minibatch_size = image_feature_curr_batch.shape[0]

        similarity_curr_batch_to_all = np.dot(image_feature_curr_batch, image_features.T).astype(np.float32)
        similarity_curr_batch_to_all[np.arange(curr_minibatch_size), np.arange(start_row_ind, end_row_ind)] = 0
        similarity_curr_batch_to_all = similarity_curr_batch_to_all > similarity_threshold

        # zero out all removals from previous batches
        if len(feature_inds_to_drop) > 0:
            similarity_curr_batch_to_all[:,np.array(feature_inds_to_drop)] = 0

        # go over the self similarity matrix rows and determine which indices should be removed
        for curr_batch_row_ind in range(curr_minibatch_size):
            if similarity_curr_batch_to_all[curr_batch_row_ind,:].sum() > 0:
                full_features_row = start_row_ind + curr_batch_row_ind
                feature_inds_to_drop.append(full_features_row)
                # zero out the column of the removed duplicate (so that it's twins won't be removed as well)
                similarity_curr_batch_to_all[:,full_features_row] = 0

    num_to_remove = len(feature_inds_to_drop)
    message_string = 'from the folder "%s" (contains %d images) \nthere will be removed %d near-duplicates (%.1f%s of images)'
    print('----------------------------------------')
    print(message_string %(base_image_folder, total_num_samples, num_to_remove, 100 * (num_to_remove / total_num_samples), '%'))
    print('----------------------------------------')

    # remove the files
    for k in feature_inds_to_drop:
        curr_image_filename = image_filename_map[k]
        curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]
        curr_features_dict_filename = os.path.join(features_folder, curr_sample_name + '.pickle')

        os.remove(curr_image_filename)
        os.remove(curr_features_dict_filename)


def delete_near_duplicates(base_image_folder, models_to_use=['CLIP_ViTL_14@336','CLIP_ResNet50x64'], similarity_threshold=0.99, minibatch_size=10_000):
    # this function does not assume "proper" folder stucture, but will create it and calculate features if necessary

    features_folder = os.path.join(base_image_folder, 'pretrained_features')

    # collect the requested features to calculate near duplication based on
    image_features, image_filename_map = extract_and_collect_pretrained_features(base_image_folder, models_to_use=models_to_use, nromalize_features=True)
    similarity_threshold = len(models_to_use) * similarity_threshold

    total_num_samples = image_features.shape[0]
    num_batches = np.ceil(total_num_samples / minibatch_size).astype(int)

    feature_inds_to_drop = []

    end_row_ind = 0
    for batch_ind in range(num_batches):
        start_row_ind = end_row_ind
        end_row_ind = min(start_row_ind + minibatch_size, total_num_samples)
        image_feature_curr_batch = image_features[start_row_ind:end_row_ind]
        curr_minibatch_size = image_feature_curr_batch.shape[0]

        similarity_curr_batch_to_all = np.dot(image_feature_curr_batch, image_features.T).astype(np.float32)
        similarity_curr_batch_to_all[np.arange(curr_minibatch_size), np.arange(start_row_ind, end_row_ind)] = 0
        similarity_curr_batch_to_all = similarity_curr_batch_to_all > similarity_threshold

        # zero out all removals from previous batches
        if len(feature_inds_to_drop) > 0:
            similarity_curr_batch_to_all[:,np.array(feature_inds_to_drop)] = 0

        # go over the self similarity matrix rows and determine which indices should be removed
        for curr_batch_row_ind in range(curr_minibatch_size):
            if similarity_curr_batch_to_all[curr_batch_row_ind,:].sum() > 0:
                full_features_row = start_row_ind + curr_batch_row_ind
                feature_inds_to_drop.append(full_features_row)
                # zero out the column of the removed duplicate (so that it's twins won't be removed as well)
                similarity_curr_batch_to_all[:,full_features_row] = 0

    num_to_remove = len(feature_inds_to_drop)
    message_string = 'from the folder "%s" (contains %d images) \nthere will be removed %d near-duplicates (%.1f%s of images)'
    print('----------------------------------------')
    print(message_string %(base_image_folder, total_num_samples, num_to_remove, 100 * (num_to_remove / total_num_samples), '%'))
    print('----------------------------------------')

    # remove the files
    for k in feature_inds_to_drop:
        curr_image_filename = image_filename_map[k]
        curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]
        curr_features_dict_filename = os.path.join(features_folder, curr_sample_name + '.pickle')

        os.remove(curr_image_filename)
        os.remove(curr_features_dict_filename)



#%% testing function of basic functionality of the module

def main():

    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        repo_dir = '/home/visual_taste_approximator/'

    os.chdir(repo_dir)

    #%% script inputs

    images_base_folder = os.path.join(repo_dir, 'images_to_label')
    positive_folder = os.path.join(repo_dir, 'positively_labeled')
    negative_folder = os.path.join(repo_dir, 'negatively_labeled')

    #%% remove duplicates from the folders

    models_to_use = ['CLIP_ViTL_14@336', 'CLIP_ViTL_14', 'CLIP_ResNet50x64']
    similarity_threshold = 0.99

    delete_near_duplicates(images_base_folder, models_to_use=models_to_use, similarity_threshold=similarity_threshold)
    delete_near_duplicates(positive_folder, models_to_use=models_to_use, similarity_threshold=similarity_threshold)
    delete_near_duplicates(negative_folder, models_to_use=models_to_use, similarity_threshold=similarity_threshold)

    #%% extract CLIP features from all images in the provided images folders

    models_to_use = ['CLIP_ViTL_14@336', 'CLIP_ViTL_14', 'CLIP_ResNet50x64']
    nromalize_features = True

    CLIP_image_features_to_label, image_filename_map_to_label = extract_and_collect_pretrained_features(images_base_folder, models_to_use=models_to_use, nromalize_features=nromalize_features)
    CLIP_image_features_positive, image_filename_map_positive = extract_and_collect_pretrained_features(positive_folder, models_to_use=models_to_use, nromalize_features=nromalize_features)
    CLIP_image_features_negative, image_filename_map_negative = extract_and_collect_pretrained_features(negative_folder, models_to_use=models_to_use, nromalize_features=nromalize_features)


    #%% train classfifer based on already labeled images and features

    n_cols = 100
    n_rows = 600
    n_neighbors = 5

    log_reg_C = 3.0
    w = [0.5, 0.5]

    X = np.concatenate((CLIP_image_features_positive, CLIP_image_features_negative))
    y = np.concatenate((np.ones((CLIP_image_features_positive.shape[0], 1)), np.zeros((CLIP_image_features_negative.shape[0], 1))))

    num_total_samples = y.shape[0]

    if num_total_samples > 0:

        reshuffled_inds = np.random.permutation(num_total_samples)
        X = X[reshuffled_inds]
        y = y[reshuffled_inds]

        # define and train model
        train_fraction = 0.7
        train_inds = np.arange(int(train_fraction * num_total_samples))
        valid_inds = np.arange(int(train_fraction * num_total_samples), num_total_samples)

        kNN_classfier = Compressed_kNN(n_cols=n_cols, n_rows=n_rows, n_neighbors=n_neighbors, whiten=False)
        kNN_classfier.fit(X[train_inds], y[train_inds])

        LDA_classfier = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
        LDA_classfier = LogisticRegression(C=log_reg_C, class_weight=[0.5, 0.5], penalty='l2', l1_ratio=None)
        LDA_classfier.fit(X[train_inds], y[train_inds][:,0])

        # evaluate performace - kNN
        y_train_hat_kNN = kNN_classfier.predict(X[train_inds])
        y_valid_hat_kNN = kNN_classfier.predict(X[valid_inds])

        train_Acc_kNN = ((y_train_hat_kNN > 0.5) == y[train_inds]).mean()
        valid_Acc_kNN = ((y_valid_hat_kNN > 0.5) == y[valid_inds]).mean()

        train_RMSE_kNN = np.sqrt(((y_train_hat_kNN - y[train_inds]) ** 2).mean())
        valid_RMSE_kNN = np.sqrt(((y_valid_hat_kNN - y[valid_inds]) ** 2).mean())

        print('kNN Accuracy (train, valid) = (%.5f, %.5f)' %(train_Acc_kNN, valid_Acc_kNN))
        print('kNN RMSE     (train, valid) = (%.5f, %.5f)' %(train_RMSE_kNN, valid_RMSE_kNN))

        # evaluate performace - LDA
        y_train_hat_LR = LDA_classfier.predict_proba(X[train_inds])[:,1]
        y_valid_hat_LR = LDA_classfier.predict_proba(X[valid_inds])[:,1]

        train_Acc_LR = ((y_train_hat_LR > 0.5) == y[train_inds][:,0]).mean()
        valid_Acc_LR = ((y_valid_hat_LR > 0.5) == y[valid_inds][:,0]).mean()

        train_RMSE_LR = np.sqrt(((y_train_hat_LR - y[train_inds][:,0]) ** 2).mean())
        valid_RMSE_LR = np.sqrt(((y_valid_hat_LR - y[valid_inds][:,0]) ** 2).mean())

        print('LR  Accuracy (train, valid) = (%.5f, %.5f)' %(train_Acc_LR, valid_Acc_LR))
        print('LR  RMSE     (train, valid) = (%.5f, %.5f)' %(train_RMSE_LR, valid_RMSE_LR))

        # train again on the full dataset
        kNN_classfier = Compressed_kNN(n_cols=n_cols, n_rows=n_rows, n_neighbors=n_neighbors, whiten=False)
        kNN_classfier.fit(X, y)
        LR_classfier = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
        LR_classfier.fit(X, y)

        # make prediction on full dataset
        y_hat_kNN = kNN_classfier.predict(X)
        y_hat_LR = LDA_classfier.predict_proba(X)[:,1]
        predicted_prob_kNN = kNN_classfier.predict(CLIP_image_features_to_label)[:,0]
        predicted_prob_LR = LR_classfier.predict_proba(CLIP_image_features_to_label)[:,1]

        yscale = 'log'
        yscale = 'linear'

        predicted_prob = w[0] * predicted_prob_kNN + w[1] * predicted_prob_LR

        bins = np.linspace(0,1,100)

        plt.close('all')
        plt.figure(figsize=(30,12))
        plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,hspace=0.22,wspace=0.15)

        plt.subplot(3,3,1); plt.title('valid subset (kNN)')
        plt.hist(y_valid_hat_kNN[y[valid_inds] == 0], bins=bins, color='red', alpha=0.8)
        plt.hist(y_valid_hat_kNN[y[valid_inds] == 1], bins=bins, color='green', alpha=0.8)
        plt.yscale(yscale)
        plt.subplot(3,3,4); plt.title('full dataset (kNN)')
        plt.hist(y_hat_kNN[y == 0], bins=bins, color='red', alpha=0.8)
        plt.hist(y_hat_kNN[y == 1], bins=bins, color='green', alpha=0.8)
        plt.yscale(yscale)

        plt.subplot(3,3,7)
        plt.hist(predicted_prob_kNN, bins=bins, color='blue'); plt.title('query dataset that will be labeled (kNN)')
        plt.yscale(yscale)


        plt.subplot(3,3,2); plt.title('valid subset (LR)')
        plt.hist(y_valid_hat_LR[y[valid_inds][:,0] == 0], bins=bins, color='red', alpha=0.8)
        plt.hist(y_valid_hat_LR[y[valid_inds][:,0] == 1], bins=bins, color='green', alpha=0.8)
        plt.yscale(yscale)

        plt.subplot(3,3,5); plt.title('full dataset (LR)')
        plt.hist(y_hat_LR[y[:,0] == 0], bins=bins, color='red', alpha=0.8)
        plt.hist(y_hat_LR[y[:,0] == 1], bins=bins, color='green', alpha=0.8)
        plt.yscale(yscale)

        plt.subplot(3,3,8)
        plt.hist(predicted_prob_LR, bins=bins, color='blue'); plt.title('query dataset that will be labeled (LR)')
        plt.yscale(yscale)


        plt.subplot(3,3,3); plt.title('valid subset (joined)')
        plt.hist(w[0] * y_valid_hat_kNN[y[valid_inds] == 0] + w[1] * y_valid_hat_LR[y[valid_inds][:,0] == 0], bins=bins, color='red', alpha=0.8)
        plt.hist(w[0] * y_valid_hat_kNN[y[valid_inds] == 1] + w[1] * y_valid_hat_LR[y[valid_inds][:,0] == 1], bins=bins, color='green', alpha=0.8)
        plt.yscale(yscale)

        plt.subplot(3,3,6); plt.title('full dataset (joined)')
        plt.hist(w[0] * y_hat_kNN[y == 0] + w[1] * y_hat_LR[y[:,0] == 0], bins=bins, color='red', alpha=0.8)
        plt.hist(w[0] * y_hat_kNN[y == 1] + w[1] * y_hat_LR[y[:,0] == 1], bins=bins, color='green', alpha=0.8)
        plt.yscale(yscale)

        plt.subplot(3,3,9)
        plt.hist(predicted_prob, bins=bins, color='blue'); plt.title('query dataset that will be labeled (joined)')
        plt.yscale(yscale)

        plt.figure(figsize=(16,8))
        plt.subplots_adjust(left=0.07,right=0.95,bottom=0.07,top=0.95,hspace=0.22,wspace=0.15)
        plt.subplot(1,2,1); plt.title('valid subset (two predictions)')
        plt.scatter(x=y_valid_hat_kNN[y[valid_inds] == 0], y=y_valid_hat_LR[y[valid_inds][:,0] == 0], color='red', alpha=0.9)
        plt.scatter(x=y_valid_hat_kNN[y[valid_inds] == 1], y=y_valid_hat_LR[y[valid_inds][:,0] == 1], color='green', alpha=0.9)
        plt.xlabel('kNN'); plt.ylabel('LR')

        plt.subplot(1,2,2); plt.title('full dataset (two predictions)')
        plt.scatter(x=y_hat_kNN[y == 0], y=y_hat_LR[y[:,0] == 0], color='red', alpha=0.7)
        plt.scatter(x=y_hat_kNN[y == 1], y=y_hat_LR[y[:,0] == 1], color='green', alpha=0.7)
        plt.xlabel('kNN'); plt.ylabel('LR')


if __name__ == "__main__":
    main()
