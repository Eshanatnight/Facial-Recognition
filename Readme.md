# Facial Recognition Pipeline with Tensorflow

## Dataset

    # Download lfw dataset
    curl -O http://vis-www.cs.umass.edu/lfw/lfw.tgz
    tar -xzvf lfw.tgz

or

run this command in the current directory

    # Download lfw dataset
    scripts\getDataset.bat

The data set used to be in the repository, but it has been removed. Because of LFS issues. get it by running the command yourself.

### Detect, Crop & Align with Dlib

After creating the environment, we begin preprocessing.

First download *dlib’s* face landmark predictor.

Just run this:

    curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

or

run this script in the current directory

    scripts\getLandmarkPredictor.bat


We’ll use this face landmark predictor to find the location of the inner eyes and bottom lips of a face in an image. These coordinates will be used to center align the image.

***Thank You Carnegie Mellon University***

## Preprocessing

Next, we'll create a preprocessor for the dataset.
This file will read each image into memory, attempt to find the largest face,
center align, and write the file to output.

If a face cannot be found in the image, *logging api* will display to console with the filename.

As each image can be processed independently, python’s multiprocessing can be used to
process an image on each available cpu core so fast!

    python preprocess.py --input-dir Data --output-dir output/intermediate --crop-dim 180

Using Dlib, we detected the largest face in an image and aligned the center of the face
by the inner eyes and bottom lip. This alignment is a method for standardizing
each image for use as feature input.

## Creating Embeddings in Tensorflow

Now that we've preprocessed the data, we’ll generate vector embeddings
of each identity. These embeddings can then be used as input to a classification,
regression or clustering task.

## Load the Embeddings

Below, we’ll utilize Tensorflow’s queue api to load the preprocessed images in parallel.
By using queues, images can be loaded in parallel using multi-threading.
When using a GPU, this allows image preprocessing to be performed on CPU,
while matrix multiplication is performed on GPU.

Do note the tensorflow needs the correct compatibility version.

## Train the Classifier

With the input queue squared away, we’ll move on to creating the embeddings.

First, you’ll load the images from the queue you created.
While training, we’ll apply preprocessing to the image. This preprocessing will add
random transformations to the image, creating more images to train on.

These images will be fed in a batch size of 128 into the model.
This model will return a 128 dimensional embedding for each image,
returning a 128 x 128 matrix for each batch.

After these embeddings are created,
we’ll use them as feature inputs into a scikit-learn’s SVM classifier to train on each identity.
Identities with less than 10 images will be dropped.
This parameter is tunable from command-line.

    # this is a lengthy process will take a while
    python --input-dir output/intermediate --model-path etc/20170511-185253/20170511-185253.pb --classifier-path output/classifier.pkl --num-threads 16 --num-epochs 25 --min-num-images-per-class 10 --is-train

## Evaluate the Results

Now with the trained classifier, we’ll evaluate the results.
Feeding it new images it has not trained on, and removing the ``is_train`` flag,
evaluates the classifier.

    python train.py --input-dir output/intermediate --model-path etc/20170511-185253/20170511-185253.pb --classifier-path output/classifier.pkl --num-threads 16 --num-epochs 5 --min-num-images-per-class 10

The result that we see is an accuracy of around 85%. That is training at 25 epochs with a batch size of 128.
And testing at 5 epochs with a batch size of 128.

## References

[Openface](https://github.com/cmusatyalab/openface/blob/master/openface/align_dlib.py)