# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# Goal:
Write a software pipeline to detect vehicles in a video. 

Steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize features
* Split training and testing data
* Train a classifier using Linear SVC classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Create heatmap of recurring detections frame by frame and remove false positives by thresholding number of windows found 
* Combine multiple boxes into a single one detected for a single car
* Verify pipeline on test images
* Run pipeline on a video stream

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. 

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
