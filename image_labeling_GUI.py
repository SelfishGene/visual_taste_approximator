import os
import sys
import shutil
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from PIL import Image
from sklearn.metrics import roc_auc_score

try:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
except:
    repo_dir = '/home/visual_taste_approximator/'

os.chdir(repo_dir)

import visual_binary_classifier
from extract_pretrained_features_module import extract_and_collect_pretrained_features, delete_near_duplicates

import warnings
warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)


#%% parse all input arguments

parser = argparse.ArgumentParser()

parser.add_argument("--test_folder", type=str, default="images_to_label",
                    help="path to folder to classify")

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


# folder_to_label = os.path.join(repo_dir, 'images_to_label')
folder_to_label = args.test_folder

# positive_output_folder = os.path.join(repo_dir, 'positively_labeled')
# negative_output_folder = os.path.join(repo_dir, 'negatively_labeled')
positive_output_folder = args.positive_folder
negative_output_folder = args.negative_folder

# task_name = 'proper_portrait_classifier'
task_name = args.model_name

# models_folder = os.path.join(repo_dir, 'models_folder')
models_folder = args.models_folder

for folder in [folder_to_label, positive_output_folder, negative_output_folder]:
    os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'pretrained_features'), exist_ok=True)


num_features_sets = args.num_features_sets

# visual features to use for classification
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
remove_duplicates = False
models_for_duplicates = ['CLIP_ViTL_14@336']
duplicates_similarity_threshold = 0.99


binary_classifier = visual_binary_classifier.VisualBinaryClassifier(models_for_features=models_for_features,
                                                                    models_for_duplicates=models_for_duplicates,
                                                                    duplicates_similarity_threshold=duplicates_similarity_threshold)


#%% train classfifer based on already labeled images (if exist) and make a prediction on test folder

try:
    perform_cross_validation = True
    show_cv_plots = False
    reset_classifier_weights = True

    y_valid_GT, y_valid_hat = binary_classifier.fit_from_folders(positive_output_folder, negative_output_folder,
                                                                 remove_duplicates=remove_duplicates,
                                                                 perform_cross_validation=perform_cross_validation,
                                                                 show_cv_plots=show_cv_plots,
                                                                 reset_classifier_weights=reset_classifier_weights)

    predicted_prob, image_filename_map_to_label = binary_classifier.predict_from_folder(folder_to_label)
    classifier_trained_on_positive = binary_classifier.n_train_samples_positive
    classifier_trained_on_negative = binary_classifier.n_train_samples_negative

    binary_classifier.save_model(models_folder, model_prefix=task_name)

except: # if the positive and negative folders are empty, still do the duplicate removal and precalculate features for all images
    if remove_duplicates:
        delete_near_duplicates(folder_to_label, models_to_use=models_for_duplicates, similarity_threshold=duplicates_similarity_threshold)
    pretrained_features, image_filename_map_to_label = extract_and_collect_pretrained_features(folder_to_label,
                                                                                               models_to_use=models_for_features)
    predicted_prob = np.zeros(pretrained_features.shape[0]) + 0.1
    classifier_trained_on_positive = 0
    classifier_trained_on_negative = 0


#%% Start Labeling GUI


positively_labeled_counter = 0
negatively_labeled_counter = 0
positively_labeled_predicted_probs = []
negatively_labeled_predicted_probs = []
was_labeled_by_user = np.zeros(predicted_prob.shape)

try:
    valid_AUC = binary_classifier.best_weights_AUC
    valid_Accuracy = binary_classifier.best_weights_Accuracy
    valid_num_samples = binary_classifier.n_train_samples_positive + binary_classifier.n_train_samples_negative
    focus_lims = [binary_classifier.best_weights_right_thresh, binary_classifier.best_weights_left_thresh]
except:
    valid_AUC = 0
    valid_Accuracy = 0
    valid_num_samples = 0
    focus_lims = [0.15,0.85]


def load_next_image():
    # bias towards predicted probability values around inside "focus_lims"
    sampling_prob = np.ones(predicted_prob.shape)
    sampling_prob[predicted_prob < focus_lims[0]] = 0.01 * sampling_prob[predicted_prob < focus_lims[0]]
    sampling_prob[predicted_prob > focus_lims[1]] = 0.01 * sampling_prob[predicted_prob > focus_lims[1]]
    sampling_prob = sampling_prob * (1 - was_labeled_by_user)
    sampling_prob /= sampling_prob.sum()
    sampled_ind = np.random.choice(np.arange(sampling_prob.shape[0]),size=1, p=sampling_prob)[0]

    curr_image_PIL = Image.open(image_filename_map_to_label[sampled_ind])
    curr_image_PIL = curr_image_PIL.convert("RGB")

    return np.array(curr_image_PIL), sampled_ind


# style parameters
text_color = 'white'
text_fontsize = 25
large_titles_fontsize = 55
GUI_background_color = '0.05'
slider_color = '0.3'
slider_background_color = '0.2'
radio_button_background_color = '0.2'
radio_button_active_color = '0.35'
radio_button_edge_color = '0.6'
button_color = '0.2'
button_hover_color = '0.3'

matplotlib.rcParams['text.color'] = text_color
matplotlib.rcParams['xtick.color'] = text_color
matplotlib.rcParams['font.size'] = text_fontsize

# Figure
plt.close('all')
fig = plt.figure(figsize=(40,30))
fig.patch.set_facecolor(GUI_background_color)

# Main image
gs_main_image = gridspec.GridSpec(nrows=1,ncols=1)
gs_main_image.update(left=0.28, bottom=0.2, right=0.72, top=0.9, wspace=0.01, hspace=0.01)
ax_main_image = plt.subplot(gs_main_image[:,:])

# main_image = load_random_image(all_images_filenames)
main_image, curr_sampled_ind = load_next_image()
curr_sample_predicted_prob = predicted_prob[curr_sampled_ind]

img_main = ax_main_image.imshow(main_image)
ax_main_image.set_axis_off()
ax_main_image.set_title('estimated probability = %.3f' %(curr_sample_predicted_prob))

# Status text
gs_main_status_text = gridspec.GridSpec(nrows=1,ncols=1)
gs_main_status_text.update(left=0.02, bottom=0.15, right=0.25, top=0.9, wspace=0.01, hspace=0.01)
ax_main_status_text = plt.subplot(gs_main_status_text[:,:])

labeling_status_format = 'Total unlabeled in folder: %d'
labeling_status_format += '\n\nTotal labeled during session: %d'
labeling_status_format += '\n\nTotal positively  labeled: %d'
labeling_status_format += '\n\nTotal negatively labeled: %d'
text_str_1 = labeling_status_format %((1 - was_labeled_by_user).sum(), was_labeled_by_user.sum(),
                                      positively_labeled_counter, negatively_labeled_counter)
classifier_status_format = '\n\n\nClassifier trained on:\n        ( -, +) = (%d, %d)'
text_str_2 = classifier_status_format %(classifier_trained_on_negative, classifier_trained_on_positive)
text_str = text_str_1 + text_str_2
status_text_box = ax_main_status_text.text(0.0, 0.98, text_str, fontsize=40, verticalalignment='top')
ax_main_status_text.set_axis_off()

# Validation histogram
gs_validation_hist = gridspec.GridSpec(nrows=1,ncols=1)
gs_validation_hist.update(left=0.77, bottom=0.75, right=0.96, top=0.9, wspace=0.01, hspace=0.01)
ax_validation_hist = plt.subplot(gs_validation_hist[:,:])


def update_valid_histograms():
    global ax_validation_hist

    try:
        if y_valid_GT.shape[0] < 200:
            valid_hist_bins = np.linspace(0,1,10)
        elif y_valid_GT.shape[0] < 1000:
            valid_hist_bins = np.linspace(0,1,25)
        else:
            valid_hist_bins = np.linspace(0,1,100)

        gs_validation_hist = gridspec.GridSpec(nrows=1,ncols=1)
        gs_validation_hist.update(left=0.77, bottom=0.75, right=0.96, top=0.9, wspace=0.01, hspace=0.01)
        ax_validation_hist = plt.subplot(gs_validation_hist[:,:])
        ax_validation_hist.set_title('classifier prediction on validation', fontsize=20)
        ax_validation_hist.hist(y_valid_hat[y_valid_GT == 0], bins=valid_hist_bins, color='red', alpha=0.8)
        ax_validation_hist.hist(y_valid_hat[y_valid_GT == 1], bins=valid_hist_bins, color='green', alpha=0.8)
        ax_validation_hist.set_xticks([0.0,0.5,1.0]); ax_validation_hist.set_xlim(-0.01,1.01);
        curr_ylim = ax_validation_hist.get_ylim()
        ax_validation_hist.plot([focus_lims[0],focus_lims[0]], curr_ylim, lw=3, color='k');
        ax_validation_hist.plot([focus_lims[1],focus_lims[1]], curr_ylim, lw=3, color='k');
        ax_validation_hist.set_ylim(curr_ylim);
    except:
        print('no valid yet')


update_valid_histograms()

# prediction on folder by classfier
gs_prediction_hist = gridspec.GridSpec(nrows=1,ncols=1)
gs_prediction_hist.update(left=0.77, bottom=0.55, right=0.96, top=0.7, wspace=0.01, hspace=0.01)
ax_prediction_hist = plt.subplot(gs_prediction_hist[:,:])

def update_prediction_histogram():
    global ax_prediction_hist

    if predicted_prob.shape[0] < 150:
        test_hist_bins = np.linspace(0,1,12)
    elif predicted_prob.shape[0] < 1000:
        test_hist_bins = np.linspace(0,1,25)
    else:
        test_hist_bins = np.linspace(0,1,100)

    gs_prediction_hist = gridspec.GridSpec(nrows=1,ncols=1)
    gs_prediction_hist.update(left=0.77, bottom=0.55, right=0.96, top=0.7, wspace=0.01, hspace=0.01)
    ax_prediction_hist = plt.subplot(gs_prediction_hist[:,:])

    ax_prediction_hist.set_title('classifier prediction on test folder', fontsize=20)
    ax_prediction_hist.hist(predicted_prob[was_labeled_by_user == 0], bins=test_hist_bins, color='blue', alpha=0.8)
    ax_prediction_hist.set_xticks([0.0,0.5,1.0]); ax_prediction_hist.set_xlim(-0.01,1.01);

    curr_ylim = ax_prediction_hist.get_ylim()
    ax_prediction_hist.plot([focus_lims[0],focus_lims[0]], curr_ylim, lw=3, color='k');
    ax_prediction_hist.plot([focus_lims[1],focus_lims[1]], curr_ylim, lw=3, color='k');
    ax_prediction_hist.set_ylim(curr_ylim);

update_prediction_histogram()

# prediction on folder by classfier
gs_empirical_hist = gridspec.GridSpec(nrows=1,ncols=1)
gs_empirical_hist.update(left=0.77, bottom=0.35, right=0.96, top=0.5, wspace=0.01, hspace=0.01)
ax_empirical_hist = plt.subplot(gs_empirical_hist[:,:])


def update_empirical_histogram():
    global ax_empirical_hist

    empirical_hist_bins = np.linspace(0,1,12)

    gs_empirical_hist = gridspec.GridSpec(nrows=1,ncols=1)
    gs_empirical_hist.update(left=0.77, bottom=0.35, right=0.96, top=0.5, wspace=0.01, hspace=0.01)
    ax_empirical_hist = plt.subplot(gs_empirical_hist[:,:])

    ax_empirical_hist.set_title('classifier prediction during session (empirical)', fontsize=20)
    h_neg = ax_empirical_hist.hist(negatively_labeled_predicted_probs, bins=empirical_hist_bins, color='red', alpha=0.8)
    h_pos = ax_empirical_hist.hist(positively_labeled_predicted_probs, bins=empirical_hist_bins, color='green', alpha=0.8)
    ax_empirical_hist.set_xticks([0.0,0.5,1.0]); ax_empirical_hist.set_xlim(-0.01,1.01);
    curr_ylim = [0, max(1, np.concatenate((h_neg[0], h_pos[0])).max())]
    ax_empirical_hist.plot([focus_lims[0],focus_lims[0]], curr_ylim, lw=3, color='k');
    ax_empirical_hist.plot([focus_lims[1],focus_lims[1]], curr_ylim, lw=3, color='k');
    ax_empirical_hist.set_ylim(curr_ylim);


update_empirical_histogram()

gs_empirical_eval_status_text = gridspec.GridSpec(nrows=1,ncols=1)
gs_empirical_eval_status_text.update(left=0.77, bottom=0.28, right=0.96, top=0.33, wspace=0.01, hspace=0.01)
ax_empirical_eval_status_text = plt.subplot(gs_empirical_eval_status_text[:,:])

evaluation_status_format_str = '(Acc, AUC) = (%.3f, %.3f), N samples = %d'
validation_eval_status_format = '\nPast cross valid: ' + evaluation_status_format_str
empirical_eval_status_format = '\nCurrent session: ' + evaluation_status_format_str
evaluation_text_str = empirical_eval_status_format %(0, 0, 0)
evaluation_text_str = ''
evaluation_text_str = evaluation_text_str + validation_eval_status_format %(valid_Accuracy, valid_AUC, valid_num_samples)
evaluation_text_str = evaluation_text_str + empirical_eval_status_format %(0, 0, 0)
empirical_text_box = ax_empirical_eval_status_text.text(0.0, 0.99, evaluation_text_str, fontsize=18, verticalalignment='top')
ax_empirical_eval_status_text.set_axis_off()


def update_empirical_test_eval():
    y_hat = np.array(negatively_labeled_predicted_probs + positively_labeled_predicted_probs)
    y_GT = np.concatenate((np.zeros(len(negatively_labeled_predicted_probs)), np.ones(len(positively_labeled_predicted_probs))))
    try:
        empirical_Accuracy = ((y_hat > 0.5) == y_GT).mean()
        empirical_AUC = roc_auc_score(y_GT, y_hat)
        empirical_num_samples = y_GT.shape[0]
    except:
        empirical_Accuracy = 0.0
        empirical_AUC = 0.0
        empirical_num_samples = 0

    evaluation_text_str = ''
    evaluation_text_str = evaluation_text_str + validation_eval_status_format %(valid_Accuracy, valid_AUC, valid_num_samples)
    evaluation_text_str = evaluation_text_str + empirical_eval_status_format %(empirical_Accuracy, empirical_AUC, empirical_num_samples)
    empirical_text_box.set_text(evaluation_text_str)
    fig.canvas.draw_idle()


update_empirical_test_eval()


def update_status_text():
    text_str_1 = labeling_status_format %((1 - was_labeled_by_user).sum(), was_labeled_by_user.sum(),
                                          positively_labeled_counter, negatively_labeled_counter)
    text_str_2 = classifier_status_format %(classifier_trained_on_negative, classifier_trained_on_positive)
    text_str = text_str_1 + text_str_2
    status_text_box.set_text(text_str)
    fig.canvas.draw_idle()


def upload_next_image(event):
    global main_image, curr_sampled_ind, was_labeled_by_user, curr_sample_predicted_prob

    if was_labeled_by_user.mean() == 1:
        plt.close('all')
    else:
        main_image, curr_sampled_ind = load_next_image()
        curr_sample_predicted_prob = predicted_prob[curr_sampled_ind]

        img_main.set_data(main_image)
        fig.canvas.draw_idle()
        ax_main_image.set_title('estimated probability = %.3f' %(curr_sample_predicted_prob))

        update_status_text()


upload_next_ax = plt.axes([0.425, 0.05, 0.15, 0.1])
upload_next_button = Button(upload_next_ax, 'Next Image\n\n(press "2")', color=button_color, hovercolor=button_hover_color)
upload_next_button.on_clicked(upload_next_image)


def get_images_features_folders(base_image_folder):
    images_folder = os.path.join(base_image_folder, 'images')
    features_folder = os.path.join(base_image_folder, 'pretrained_features')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(features_folder, exist_ok=True)

    return images_folder, features_folder


def copy_image_and_features_to_dst(curr_full_image_filename_src, dest_folder):

    folder_images, folder_features = get_images_features_folders(dest_folder)

    curr_sample_name = curr_full_image_filename_src.split('/')[-1].split('.')[0]
    curr_full_features_filename_src = os.path.join(folder_to_label, 'pretrained_features', curr_sample_name + '.pickle')

    curr_sample_name_dst = curr_sample_name + '_%.5d' %(np.random.randint(99999))
    curr_sample_name_dst_image = curr_sample_name_dst + '.' + curr_full_image_filename_src.split('.')[-1]
    curr_sample_name_dst_features = curr_sample_name_dst + '.pickle'

    # copy image and features to destination folder
    shutil.copy(curr_full_image_filename_src, os.path.join(folder_images, curr_sample_name_dst_image))
    shutil.copy(curr_full_features_filename_src, os.path.join(folder_features, curr_sample_name_dst_features))

    # remove image and features from original folder
    os.remove(curr_full_image_filename_src)
    os.remove(curr_full_features_filename_src)


# Label as positive button
def label_as_positive_sample(event):
    global positively_labeled_counter
    positively_labeled_counter += 1

    curr_full_image_filename = image_filename_map_to_label[curr_sampled_ind]
    copy_image_and_features_to_dst(curr_full_image_filename, positive_output_folder)

    # global positively_labeled_counter
    # positively_labeled_counter += 1
    # image_filename = positive_output_folder + 'images/sample_%0.8d.png' %(np.random.randint(999999))
    # Image.fromarray(main_image, 'RGB').save(image_filename)

    upload_next_image(event)


positive_sample_botton_ax = plt.axes([0.585, 0.05, 0.15, 0.1])
positive_sample_botton = Button(positive_sample_botton_ax, 'Positive Sample\n\n(press "3")',
                                color=button_color, hovercolor=button_hover_color)
positive_sample_botton.on_clicked(label_as_positive_sample)


# Label as negative button
def label_as_negative_sample(event):
    global negatively_labeled_counter
    negatively_labeled_counter += 1

    curr_full_image_filename = image_filename_map_to_label[curr_sampled_ind]
    copy_image_and_features_to_dst(curr_full_image_filename, negative_output_folder)

    # global negatively_labeled_counter
    # negatively_labeled_counter += 1
    # image_filename = negative_output_folder + 'images/sample_%0.8d.png' %(np.random.randint(999999))
    # Image.fromarray(main_image, 'RGB').save(image_filename)

    upload_next_image(event)


negative_sample_botton_ax = plt.axes([0.265, 0.05, 0.15, 0.1])
negative_sample_botton = Button(negative_sample_botton_ax, 'Negative Sample\n\n(press "1")',
                                color=button_color, hovercolor=button_hover_color)
negative_sample_botton.on_clicked(label_as_negative_sample)


# retrain classifier right now button
def retrain_classifier(event):
    global classifier_trained_on_positive, classifier_trained_on_negative
    global positively_labeled_predicted_probs, negatively_labeled_predicted_probs
    global predicted_prob, image_filename_map_to_label, binary_classifier, y_valid_GT, y_valid_hat
    global valid_AUC, valid_Accuracy, valid_num_samples, was_labeled_by_user

    # re-train classifier and perform cross validation
    y_valid_GT, y_valid_hat = binary_classifier.fit_from_folders(positive_output_folder, negative_output_folder,
                                                                 remove_duplicates=remove_duplicates,
                                                                 perform_cross_validation=perform_cross_validation,
                                                                 show_cv_plots=show_cv_plots,
                                                                 reset_classifier_weights=reset_classifier_weights)

    print('------------------------------------------')
    print('finished re-training classifier')
    print('------------------------------------------')

    # make a new prediction on the
    predicted_prob, image_filename_map_to_label = binary_classifier.predict_from_folder(folder_to_label)
    classifier_trained_on_positive = binary_classifier.n_train_samples_positive
    classifier_trained_on_negative = binary_classifier.n_train_samples_negative
    valid_AUC = binary_classifier.best_weights_AUC
    valid_Accuracy = binary_classifier.best_weights_Accuracy
    valid_num_samples = binary_classifier.n_train_samples_positive + binary_classifier.n_train_samples_negative

    # save the classifier for later usage
    binary_classifier.save_model(models_folder, model_prefix=task_name)

    # clear the empirical predictions
    positively_labeled_predicted_probs = []
    negatively_labeled_predicted_probs = []
    was_labeled_by_user = np.zeros(predicted_prob.shape)

    # update the GUI
    update_status_text()
    update_valid_histograms()
    update_prediction_histogram()
    update_empirical_histogram()
    update_empirical_test_eval()
    upload_next_image(event)


retrain_classifier_botton_ax = plt.axes([0.02, 0.05, 0.225, 0.15])
retrain_classifier_botton = Button(retrain_classifier_botton_ax, 'Re-train Classifier now\n\n(press "t")',
                                   color=button_color, hovercolor=button_hover_color)
retrain_classifier_botton.on_clicked(retrain_classifier)


# left and right threshold sliders that update the sampling limits (focus_lims)
def update_focus_lims(val):
    global focus_lims

    # set the focus lims
    focus_lims[0] = left_threshold_slider.val
    focus_lims[1] = right_threshold_slider.val

    # update the plots that need to redraw the focus lims
    update_valid_histograms()
    update_prediction_histogram()
    update_empirical_histogram()


left_threshold_slider_ax = plt.axes([0.769, 0.24, 0.19, 0.025], facecolor=slider_background_color)
left_threshold_slider = Slider(left_threshold_slider_ax, label='Left thresh',
                               color=slider_color, valmin=0.0, valmax=1.0, valinit=focus_lims[0])
left_threshold_slider.on_changed(update_focus_lims)

right_threshold_slider_ax = plt.axes([0.769, 0.20, 0.19, 0.025], facecolor=slider_background_color)
right_threshold_slider = Slider(right_threshold_slider_ax, label='right thresh',
                                color=slider_color, valmin=0.0, valmax=1.0, valinit=focus_lims[1])
right_threshold_slider.on_changed(update_focus_lims)


# keyboard shortcuts
def on_press(event):
    global was_labeled_by_user, negatively_labeled_predicted_probs, positively_labeled_predicted_probs
    global ax_empirical_hist

    # print('pressed', event.key)
    sys.stdout.flush()
    if event.key == 't':
        retrain_classifier(event)
    if event.key == '1':
        was_labeled_by_user[curr_sampled_ind] = 1
        negatively_labeled_predicted_probs.append(predicted_prob[curr_sampled_ind])
        label_as_negative_sample(event)
        update_empirical_histogram()
        update_empirical_test_eval()
        update_prediction_histogram()
    elif event.key == '2':
        upload_next_image(event)
    elif event.key == '3':
        was_labeled_by_user[curr_sampled_ind] = 1
        positively_labeled_predicted_probs.append(predicted_prob[curr_sampled_ind])
        label_as_positive_sample(event)
        update_empirical_histogram()
        update_empirical_test_eval()
        update_prediction_histogram()


fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()
