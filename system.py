"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
from collections import Counter
import operator
import enchant
import cv2


def divergence(class1, class2):
    """compute a vector of 1-D divergences

    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2

    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12


def reduce_dimensions_train(feature_vectors_full, model):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    upper_letters = []
    lower_letters = []
    div_matrix = np.empty([1, 40])

    # reduce features down to 40
    reduction_40 = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), model['v'])
    print('pca reduction shape: ', reduction_40.shape)

    # get training labels from dictionary
    labels_train = np.array(model['labels_train'])

    # get all upper case letter rows from reduced data
    for i in range(65, 91):
        upper_letters.append(reduction_40[labels_train[:] == chr(i), :])
    upper_letters = np.array(upper_letters)
    print(upper_letters[0].shape)

    # get all lower case letter rows from reduced data
    for i in range(97, 123):
        lower_letters.append(reduction_40[labels_train[:] == chr(i), :])
    lower_letters = np.array(lower_letters)
    print(lower_letters.shape)

    # concatenate upper and lower case letters data into one list of arrays for each letter
    all_letters = np.concatenate((upper_letters, lower_letters), axis=0)
    print(all_letters[1].shape)

    # get divergence for every pair of letters (do I sum or..?!)
    for i in range(1, len(all_letters)):
        if i < 52:
            for j in range(i+1, len(all_letters)):
                if all_letters[i].shape[0] >= 20 and all_letters[j].shape[0] >= 20:
                    temp_div = divergence(all_letters[i], all_letters[j])
                    div_matrix = np.vstack((div_matrix, temp_div))
    print(div_matrix)

    # sort features by largest divergence, select 10 indexes of largest as features to use from pca reduced data
    sorted_indexes = np.argsort(-div_matrix, axis=1)[:, 0:10]
    print('sorted index shape: ', sorted_indexes.shape, sorted_indexes)
    sorted_indexes = sorted_indexes.flatten()
    print('flattened index shape: ', sorted_indexes.shape, sorted_indexes)
    count_indexes = np.bincount(sorted_indexes)
    print('count index: ', count_indexes.shape, count_indexes)
    final_reduction_row_indexes = np.argsort(-count_indexes)[0:10]
    # print(final_reduction)

    final_reduction_row_indexes = np.array(final_reduction_row_indexes)
    model['row_indexes'] = final_reduction_row_indexes.tolist()
    print(final_reduction_row_indexes)

    # adata = reduction_40[labels_train[:] == chr(65), :]
    # bdata = reduction_40[labels_train[:] == chr(66), :]
    # div = divergence(adata, bdata)
    # sorted = np.argsort(-div)
    # index_rows = sorted[0:10]
    # print(index_rows)

    reduction = reduction_40[:, final_reduction_row_indexes]
    print(reduction.shape)

    return reduction


def reduce_dimensions_test(feature_vectors_full, model):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    row_indexes = np.array(model['row_indexes'])
    v = np.array(model['v'])
    train_mean = np.array(model['train_mean'])

    reduction = np.dot((feature_vectors_full - train_mean), v[:, row_indexes])
    # print(reduction.shape)

    return reduction

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    print(bbox_size)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)
    # print(fvectors_train_full.shape, fvectors_train_full)

    model_data = dict()
    model_data['train_mean'] = np.mean(fvectors_train_full).tolist()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')

    covx = np.cov(fvectors_train_full, rowvar=False)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
    v = np.fliplr(v)
    model_data['v'] = v.tolist()

    fvectors_train = reduce_dimensions_train(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()

    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    # kernel = np.ones((2, 2), np.float32) / 4
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    # images_test_final = []
    # for image in images_test:
    #     median = cv2.medianBlur(image, 5)
    #     images_test_final.append(median)
    # images_test_final = np.array(images_test_final)
    # print(images_test.shape)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions_test(fvectors_test, model)
    return fvectors_test_reduced


def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.

        parameters:

        page - 2d array, each row is a feature vector to be classified
        labels - the output classification label for each feature vector
        bboxes - 2d array, each row gives the 4 bounding box coords of the character
        model - dictionary, stores the output of the training stage
        """
    # word_str = ''
    # char_list = [",", ".", "!", ";", ":"]
    # full_dict = enchant.Dict("en_UK")
    # print(labels[10])
    #
    # for i in range(len(bboxes)-1):
    #     if labels[i] not in char_list:
    #         word_str += labels[i]
    #         space = (bboxes[i+1][0] - bboxes[i][2])
    #         if space > 6:
    #             if not full_dict.check(word_str):
    #                 split_word = False
    #                 for j in range(1, len(word_str)):
    #                     left = word_str[:j]
    #                     right = word_str[j:]
    #                     if full_dict.check(left) and full_dict.check(right):
    #                         split_word = True
    #                 if not split_word:
    #                     all_sugg = full_dict.suggest(word_str)
    #                     reduced_sugg = [sugg for sugg in all_sugg if len(sugg) == len(word_str)]
    #                     print(reduced_sugg)
    #                     correction = ""
    #                     correction_score = 3000
    #                     for sugg_word in reduced_sugg:
    #                         diff = hamdist(word_str, sugg_word)
    #                         if diff < correction_score:
    #                             correction = sugg_word
    #                             correction_score = diff
    #                     print(word_str)
    #                     print(correction)
    #                     if correction != "":
    #                         for m in range(len(word_str)-1):
    #                             if word_str[m] != correction[m]:
    #                                 labels[i - ((len(word_str)-1) - m)] = correction[m]
    #             word_str = ''

    return labels


def hamdist(str1, str2):
    """Count the # of differences between equal length strings str1 and str2"""

    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs


def get_neighbour_labels(nearest):

    counts = [[]]

    for neighbours in nearest:
        i = 0
        for label in neighbours:
            if label in counts:
                counts[i][label] += 1
            else:
                counts[i][label] = 1
        i += 1

    counts = np.array(counts)



    return labels


def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """

    """Perform nearest neighbour classification."""
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    # Use all feature is no feature parameter has been supplied
    if fvectors_train is None:
        fvectors_train = np.arange(0, fvectors_train.shape[1])

    # Super compact implementation of nearest neighbour
    x = np.dot(page, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argsort(-dist, axis=1)[:, 0:5]
    labels = []
    for i in range(len(nearest)):
        counted_labels = Counter(labels_train[nearest[i]])
        labels.append(max(counted_labels, key=counted_labels.get))
    labels = np.array(labels)
    # count_labels = [np.bincount(nearest[i]) for i in range(nearest.shape[0])]
    # print(count_labels[0])
    # count_labels = np.array(count_labels)
    # print(count_labels)

    return labels
