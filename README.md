# OCR Using PCA and KNN

This is a small project which performs optical character recognition using principle compoenent analysis to extract feature fectures, then uses k-nearest neighbours to classify a character in a given bounding box.

The code written by me is in `system.py`.  All other code is used for setup and evaluation purposes.

## Setup and Running

1. Ensure that you have Python 3 installed.
2. Clone this repository locally.
3. Navigate to the root project directory and install the required packages (in requirements.txt) using pip (`pip install -r requirements.txt`)
4. (optional) Run the training in order to update the 'model.json.gz' file (in '/data').
5. Run evaluation on the model using `python evaluate.py dev` (this takes a little while, be patient!).

The results will show the acccuracy of the classifier (percentage of correctly classified characters) on each page within the '/data/dev/' directory.  You can view the pages in pdf format within this directory.
