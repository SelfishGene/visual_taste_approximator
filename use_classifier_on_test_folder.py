import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
except:
    repo_dir = '/home/visual_taste_approximator/'

os.chdir(repo_dir)

import visual_binary_classifier
from extract_pretrained_features_module import delete_worst_near_duplicates


#%% parse all input arguments

parser = argparse.ArgumentParser()

parser.add_argument("--test_folder", type=str, default="images_to_label",
                    help="path to folder to classify")

parser.add_argument("--model_path", type=str,
                    default="models_folder/proper_portrait_classifier_num_samples_200_num_features_6784_2022-09-05.pickle",
                    help="path to pretrained model to use for classification")

parser.add_argument("--positive_out_folder", type=str, default="likely_positive",
                    help="path to positive output folder images")

parser.add_argument("--negative_out_folder", type=str, default="likely_negative",
                    help="path to negative output folder images")

parser.add_argument("--positive_threshold", type=float, default=0.7,
                    help="threshold above which all images in src 'test_folder' will be moved to the 'positive_out_folder'")

parser.add_argument("--negative_threshold", type=float, default=0.3,
                    help="threshold below which all images in src 'test_folder' will be moved to the 'negative_out_folder'")

parser.add_argument("--delete_src", action='store_true',
                    help="if enabled, deletes files in src 'test_folder' that are transfered to output positive and negative folders")

args = parser.parse_args()

test_folder_name = args.test_folder
dst_positive_folder_name = args.positive_out_folder
dst_negative_folder_name = args.negative_out_folder
almost_certainly_positive_thresh = args.positive_threshold
almost_certainly_negative_thresh = args.negative_threshold
delete_src = args.delete_src

#%% load already trained classifier

# model_filename = 'proper_portrait_classifier_num_samples_200_num_features_6784_2022-09-05.pickle'
# model_folder = os.path.join(repo_dir, 'models_folder')
# model_full_path = os.path.join(model_folder, model_filename)
model_full_path = args.model_path
binary_classifier = visual_binary_classifier.load_pretrained_model(model_full_path)

#%% make a prediction on all files in a new folder using the classifier

folder_to_classify = os.path.join(repo_dir, test_folder_name)

predicted_probability, image_filename_map = binary_classifier.predict_from_folder(folder_to_classify)

# plot histogram of predicted probabilties
fraction_above_085 = (predicted_probability >= 0.85).mean()
fraction_below_025 = (predicted_probability <= 0.25).mean()

plt.figure(figsize=(12,7))
plt.hist(predicted_probability, bins=np.linspace(0,1,100))
plt.title('%.1f%s are almost_certainly bad \n%.1f%s are almost_certainly good' %(100 * fraction_below_025, '%', 100 * fraction_above_085, '%'))

#%% transfer images from folder to new "almost certainly good" and "almost certainly bad" folders according to classifier


def get_images_features_folders(base_image_folder):
    images_folder = os.path.join(base_image_folder, 'images')
    features_folder = os.path.join(base_image_folder, 'pretrained_features')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(features_folder, exist_ok=True)

    return images_folder, features_folder


def copy_image_and_features_to_dst(curr_full_image_filename_src, dest_folder, delete_src=True):

    folder_images, folder_features = get_images_features_folders(dest_folder)

    curr_sample_name = curr_full_image_filename_src.split('/')[-1].split('.')[0]
    curr_full_features_filename_src = os.path.join(folder_to_classify, 'pretrained_features', curr_sample_name + '.pickle')

    curr_sample_name_dst = curr_sample_name + '_%.5d' %(np.random.randint(99999))
    curr_sample_name_dst_image = curr_sample_name_dst + '.' + curr_full_image_filename_src.split('.')[-1]
    curr_sample_name_dst_features = curr_sample_name_dst + '.pickle'

    # copy image and features to destination folder
    shutil.copy(curr_full_image_filename_src, os.path.join(folder_images, curr_sample_name_dst_image))
    shutil.copy(curr_full_features_filename_src, os.path.join(folder_features, curr_sample_name_dst_features))

    if delete_src:
        # remove image and features from original folder
        os.remove(curr_full_image_filename_src)
        os.remove(curr_full_features_filename_src)


almost_certainly_negative_folder = os.path.join(repo_dir, dst_negative_folder_name)
almost_certainly_positive_folder = os.path.join(repo_dir, dst_positive_folder_name)

os.makedirs(almost_certainly_negative_folder, exist_ok=True)
os.makedirs(almost_certainly_positive_folder, exist_ok=True)

# go over all images and transfer the relevent images to relevent folders
for k, curr_predicted_prob in enumerate(predicted_probability):
    full_image_filename = image_filename_map[k]

    if curr_predicted_prob <= almost_certainly_negative_thresh:
        copy_image_and_features_to_dst(full_image_filename, almost_certainly_negative_folder, delete_src=delete_src)

    if curr_predicted_prob >= almost_certainly_positive_thresh:
        copy_image_and_features_to_dst(full_image_filename, almost_certainly_positive_folder, delete_src=delete_src)


#%% remove near-duplicates from the folders and cross near duplicates

remove_near_duplicates = False

if remove_near_duplicates:

    similarity_threshold = 0.95
    feature_models_to_use = binary_classifier.models_for_features

    for base_image_folder in [almost_certainly_positive_folder, almost_certainly_negative_folder]:

        delete_worst_near_duplicates(base_image_folder, binary_classifier,
                                     models_to_use=feature_models_to_use,
                                     similarity_threshold=similarity_threshold)


#%%

