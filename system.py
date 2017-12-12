"""OCR classification system.

This code trains and tests a classifier from pages from a book containing
bounding boxes around each letter.

version: v2.0
"""
import numpy as np
import utils.utils as utils
from collections import Counter
from scipy import ndimage, linalg, math
import enchant


def divergence(class1, class2):
    """compute a vector of 1-D divergences

    Params:
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
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12


def reduce_dimensions_train(feature_vectors_full, model):
    """
    Reduces training data to 40 dimensions using PCA then selects the best
    10 of these using divergence, reducing the data to 10 dimensions

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    upper_letters = []
    lower_letters = []
    div_matrix = np.empty([1, 40])  # matrix containing all vectors of 1D-divergences

    # centre features by mean and reduce down to 40 dimensions using PCA (eigenvectors)
    reduction_40 = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), model['v'])

    # get training labels from dictionary
    labels_train = np.array(model['labels_train'])

    # get all upper case letter rows from PCA reduced data
    # (65 to 90 are ASCII references for upper case letters)
    for i in range(65, 91):
        upper_letters.append(reduction_40[labels_train[:] == chr(i), :])
    upper_letters = np.array(upper_letters)

    # get all lower case letter rows from reduced data
    # (97 to 122 are ASCII references for lower case letters)
    for i in range(97, 123):
        lower_letters.append(reduction_40[labels_train[:] == chr(i), :])
    lower_letters = np.array(lower_letters)

    # concatenate upper and lower case letters data into one list of lists for each letter
    all_letters = np.concatenate((upper_letters, lower_letters), axis=0)

    # get divergence for every pair of letters
    for i in range(1, len(all_letters)):
        if i < 52:
            for j in range(i+1, len(all_letters)):
                if all_letters[i].shape[0] >= 20 and all_letters[j].shape[0] >= 20:
                    temp_div = divergence(all_letters[i], all_letters[j])
                    div_matrix = np.vstack((div_matrix, temp_div))  # stack divergence data into matrix

    # select 10 highest divergence indexes (the features), keeping all rows
    sorted_indexes = np.argsort(-div_matrix, axis=1)[:, 0:10]
    sorted_indexes = sorted_indexes.flatten()
    # find the 10 most common features from all divergences between all pairs of classes
    count_indexes = np.bincount(sorted_indexes)
    final_reduction_row_indexes = np.argsort(-count_indexes)[0:10]
    final_reduction_row_indexes = np.array(final_reduction_row_indexes)
    # add selected features to dictionary
    model['row_indexes'] = final_reduction_row_indexes.tolist()

    # reduce the PCA reduced data down to 10 dimensions by selecting the 10
    # best features from divergence
    reduction = reduction_40[:, final_reduction_row_indexes]

    return reduction


def reduce_dimensions_test(feature_vectors_full, model):
    """
    Reduce test data down to 10 dimensions

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    row_indexes = np.array(model['row_indexes'])  # best features to use (found in training)
    v = np.array(model['v'])
    train_mean = np.array(model['train_mean'])

    # reduce to 10 dimensions by selecting the 10 features found in training from the eigenvectors
    reduction = np.dot((feature_vectors_full - train_mean), v[:, row_indexes])

    return reduction


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """
    Reformat characters into feature vectors.

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


"""Contrast functions not used as they reduce accuracy of classifier"""
def increase_contrast_image(image, split):
    """
    Increase the contrast of an image by changing any pixel below a certain
    value to white and any pixel above this value to black

    Params:
    image - an image stored as a list of pixel values
    split - an integer to represent a pixel value where any pixel above this
            value will be set to 255, any below will be set to 0

    return - the modified image (as list of new pixel values)
    """

    for i in range(len(image)-1):
        for j in range(len(image[0])-1):
            if image[i][j] >= split:
                image[i][j] = 255  # make pixel white
            else:
                image[i][j] = 0  # make pixel black

    return image


def increase_contrast_vector(vector, split):
    """
    Increase the contrast of an vector representing an image
    by changing any feature below a certain value to 0 and
    any pixel above this value to 255

    Params:
    image - an image stored as a list of pixel values
    split - an integer to represent a pixel value where any pixel above this
            value will be set to 255, any below will be set to 0

    return - the modified image (as list of new pixel values)
    """

    for i in range(len(vector)-1):
        if vector[i] >= split:
            vector[i] = 255  # make pixel white
        else:
            vector[i] = 0  # make pixel black

    return vector


def process_training_data(train_page_names):
    """
    Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    images_train_final = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)

    # for every image, increase contrast and store these new images as
    # the images to use for training
    # for image in images_train:
    #     img_contr = increase_contrast_image(image, 150)
    #     images_train_final.append(img_contr)

    # images_train_final = np.array(images_train_final)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    # take first half of full training vectors
    fvectors_train_fhalf = fvectors_train_full[:(math.floor((fvectors_train_full.shape[0])/2)), :]
    # create random 1D array with n features to be used as noise
    np.random.seed(2)
    noise = (np.random.rand(2340) * 100).astype(int)
    # add noise to half of training data images
    fvectors_train_fhalf = np.subtract(fvectors_train_fhalf, noise)

    # any pixel below 0 (black) set to 0
    for i in range(len(fvectors_train_fhalf)-1):
        for j in range(len(fvectors_train_fhalf[0])-1):
            if fvectors_train_fhalf[i][j] < 0:
                fvectors_train_fhalf[i][j] = 0

    # for every noisy training images, apply a median filter to image to reduce
    # noise and store these new images as the images to use for training
    # (same filters applied to test images)
    fvectors_train_fhalf_final = []
    for vector in fvectors_train_fhalf:
        # img_contr = increase_contrast_vector(vector, 150) -- commented out as it reduces accuracy
        noise_red = ndimage.median_filter(vector, 3)
        fvectors_train_fhalf_final.append(noise_red)
    fvectors_train_fhalf_final = np.array(fvectors_train_fhalf_final)

    # recreate full training vectors by stacking noisy images with second half of original full training vectors
    fvectors_train_shalf = fvectors_train_full[(math.floor((fvectors_train_full.shape[0])/2)):, :]
    fvectors_train_full = np.vstack((fvectors_train_fhalf_final, fvectors_train_shalf))

    model_data = dict()
    model_data['train_mean'] = np.mean(fvectors_train_full).tolist()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')

    # use PCA to get 40 eigenvectors of covariance matrix of all training vectors
    covx = np.cov(fvectors_train_full, rowvar=False)
    N = covx.shape[0]
    w, v = linalg.eigh(covx, eigvals=(N - 40, N - 1))
    v = np.fliplr(v)
    model_data['v'] = v.tolist()

    fvectors_train = reduce_dimensions_train(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()

    return model_data


def load_test_page(page_name, model):
    """
    Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    images_test_final = []

    # for every test image, apply a median filter to image to reduce noise
    # and store these new images as the images to use for testing
    for image in images_test:
        # img_contr = increase_contrast_image(image, 150) -- commented out as reduces accuracy
        noise_red = ndimage.median_filter(image, 3)
        images_test_final.append(noise_red)
    images_test_final = np.array(images_test_final)

    fvectors_test = images_to_feature_vectors(images_test_final, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions_test(fvectors_test, model)

    return fvectors_test_reduced


def correct_errors(page, labels, bboxes, model):
    """
    Takes labels from classification, makes words out of them and spell checks the word

    Parameters:
    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage

    return - modified labels

    -- commented out as makes accuracy worse --
    """
    # word_str = ''
    # char_list = [",", ".", "!", ";", ":"]  # chars not to include in words
    # full_dict = enchant.Dict("en_UK")  # use English dictionary from 'enchant' package
    #
    # for i in range(len(bboxes)-1):
    #     if labels[i] not in char_list:
    #         word_str += labels[i]  # add label to string
    #         space = (bboxes[i+1][0] - bboxes[i][2])  # check if there is a space
    #         if space > 6:
    #             if not full_dict.check(word_str):  # check whether labels before space are a word
    #                 split_word = False  # set a flag for if letters in the word string can be split into 2 or 3 words
    #                 for j in range(1, len(word_str)-2):  # iterate through whole word checking for new split words
    #                     for k in range(j+1, len(word_str)):
    #                         left = word_str[:j]
    #                         middle = word_str[j:k]
    #                         right = word_str[k:]
    #                         if (full_dict.check(left) and full_dict.check(right) and full_dict.check(middle)) or\
    #                                 (full_dict.check(left) and full_dict.check(right)):
    #                             split_word = True
    #                 if not split_word:
    #                     all_sugg = full_dict.suggest(word_str)  # find all suggested words
    #                     # suggestions must be same length as original word
    #                     reduced_sugg = [sugg for sugg in all_sugg if len(sugg) == len(word_str)]
    #                     correction = ""
    #                     correction_score = 3000
    #                     # choose suggested word with fewest label changes
    #                     for sugg_word in reduced_sugg:
    #                         diff = hamdist(word_str, sugg_word)
    #                         if diff < correction_score:
    #                             correction = sugg_word
    #                             correction_score = diff
    #                     # iterate through each label from the original word and correct to new suggestion iff
    #                     # there are 3 or less corrections to make
    #                     if correction != "" and correction_score <= 3:
    #                         for m in range(len(word_str)-1):
    #                             if word_str[m] != correction[m] and not word_str[m].isupper():
    #                                 labels[i - ((len(word_str)-1) - m)] = correction[m]
    #             word_str = ''

    return labels


def hamdist(str1, str2):
    """
    Count the number of differences between equal length strings

    Params:
    str1 - a string to compare
    str2 - a string to compare
    """

    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs


def classify_page(page, model):
    """
    k-nearest neighbour classifier

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    x = np.dot(page, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    x /= np.outer(modtest, modtrain.transpose())  # cosine distance
    dist = x
    nearest = np.argsort(-dist, axis=1)[:, 0:5]  # get 5 nearest neighbours

    labels = []
    # get the most common label from nearest neighbours and use this as the classified label
    for i in range(len(nearest)):
        # Counter returns a dictionary of key: label and value: number of instances of this key
        counted_labels = Counter(labels_train[nearest[i]])
        labels.append(max(counted_labels, key=counted_labels.get))  # get key (label) with most common occurrence
    labels = np.array(labels)

    return labels
