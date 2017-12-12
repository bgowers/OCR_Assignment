# OCR assignment report

## Feature Extraction (Max 200 Words)

[Starting with the training data, images are turned into feature
vectors then this is split in half. The first of the images have
noise added to them, then have a median filter applied to them
to soften the noise. The reason for this is because the same
filter is applied to all test images. The noisy train images are then
stacked back with the second half of the full feature vectors. PCA
reduction is then used to reduce the vectors to contain 40 features.
From this PCA reduced data, divergence is computed for every pair
of letters in the set {A-Z,a-z}. Each divergence vector is then sorted
and the largest 10 are selected for each pair of letters. Then the
best 10 features are selected from this matrix by flattening it and
counting the 10 most common indexes. These indexes are saved
in a dictionary. The test data will then be reduced by doing PCA to
10 features by selecting the saved column indexes from the
eigenvector matrix before doing the dot product of the centred data.]

## Classifier (Max 200 Words)

[The classifier is a k-nearest neighbour classifier using the cosine
distance from the training feature vectors.  The nearest 5
neighbours are inspected, the most common label of those
neighbours is selected to classify the image being tested.]

## Error Correction (Max 200 Words)

[Error correction is commented out as my attempt reduces the
accuracy of the output labels. I tried to implement error correction
that would iterate through the bounding boxes adding each
label associated with that bounding box to a string until a gap greater
than 6 pixels was found between bounding boxes. The string is then
tested to see If it contains a word, first by checking the whole string
against a dictionary, then by splitting the word to see if two or three
words were accidentally joined. If none of these strings are contained
within the English dictionary, a list of suggested words that are the
same length as the original string is generated, the difference between
all generated words and the original word string is computed using
hamming distance (number of different letters in generated word) then
the generated word (correction) with the smallest difference is selected.
If the correction word contains 3 or fewer differences, then the original
labels are updated to the differences in the correction.]

## Performance

The percentage errors (to 1 decimal place) for the development data are
as follows:

‐ Page 1: [97.0%]
‐ Page 2: [98.2%]
‐ Page 3: [90.7%]
‐ Page 4: [73.7%]
‐ Page 5: [59.4%]
‐ Page 6: [44.6%]

## Other information (Optional, Max 100 words)

[Initially I just used PCA reduction to reduce to 10 features but this
produced poor results. Before adding noise to the training images,
I tried raising the contrast of both the test and train images by
making any pixel below a value black, otherwise white. This raised
the accuracy of the classifier slightly before adding noise to the
training images was implemented, where changing the contrast
then reduced the accuracy.]