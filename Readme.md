# Facial Recognition Pipeline with Tensorflow

## Dataset

    # Download lfw dataset
    curl -O http://vis-www.cs.umass.edu/lfw/lfw.tgz
    tar -xzvf lfw.tgz

The data set used to be in the repository, but it has been removed. Because of LFS issues. get it by running the command yourself.

## Environment Setup

We use Docker but its not required.

### Detect, Crop & Align with Dlib

After creating the environment, we begin preprocessing.

First download *dlib’s* face landmark predictor.

Just run this:

    curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

We’ll use this face landmark predictor to find the location of the inner eyes and bottom lips of a face in an image. These coordinates will be used to center align the image.

***Thank You Carnegie Mellon University***



## References

[Openface](https://github.com/cmusatyalab/openface/blob/master/openface/align_dlib.py)