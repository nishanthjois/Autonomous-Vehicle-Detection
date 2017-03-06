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




### Histogram of Color
Template mathching are not robust to changes in appearance, hence we use a better transformation method which is to compute histogram of color values, this gives locations of similar distribution a close match therefore removing sensitiviting to perfect arrangement of pixels 

We can construct histograms of the R, G, and B channels like this:
```
  rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
  ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
  bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
  ```
Results of plotting:


Disadvantage of histogram of color transformation is we are purely relying on color.

### Color spaces
Cars are more saturated than background like sky, road, etc., hence we will explore different color spaces (RGB, HSV, YCrRB) and also we will check the difference between choosen color space for car image and non car image. 

See results of various 

### Spatial binning
Raw pixel values are still quite useful to include in our feature vector in searching for cars. As it will be cumbersome to include three color channels of a full resolution image we can perform spatial binning on an image and still retain enough information to help in finding vehicles.

As we can see in the example above, even going all the way down to 32 x 32 pixel resolution, the car itself is still clearly identifiable by eye, and this means that the relevant features are still preserved at this resolution.

small_img = cv2.resize(image, (32, 32))


### HOG:
Gradients and edges gives more robust representation and captures notion of shape. 

There is an excellent tutorial here: http://www.learnopencv.com/histogram-of-oriented-gradients/ but I will list few important terms here:

In the HOG feature descriptor, the distribution (histograms) of directions of gradients (oriented gradients) are used as features. Gradients (x and y derivatives ) of an image are useful because the magnitude of gradients is large around edges and corners (regions of abrupt intensity changes) and we know that edges and corners pack in a lot more information about object shape than flat regions. 

At every pixel, the gradient has a magnitude and a direction. We will group these pixels into small cells (say 8 x8 pixels). Inside these cell will compute HOG. The x-gradient fires on vertical lines and the y-gradient fires on horizontal lines. The magnitude of gradient fires where ever there is a sharp change in intensity. None of them fire when the region is smooth. For color images, the gradients of the three channels are evaluated. The magnitude of gradient at a pixel is the maximum of the magnitude of gradients of the three channels, and the angle is the angle corresponding to the maximum gradient.

The histogram is essentially a vector (or an array ) of 9 bins (numbers) corresponding to angles 0, 20, 40, 60 … 160. Gradient samples are distributed into these bins and summed up. 

Each pixel in the image gets a vote on which histogram bin it belongs based on the gradient direction at that position, but the weight of that vote depends on the gradient magnitude at that pixel. When we do this for all the cells we see a representation of the original strucutre image, this can be used as a signature for a given shape.

The scikit-image package has a built in function to extract Histogram of Oriented Gradient features:

The scikit-image hog() function takes in a single color channel or grayscaled image as input, as well as various parameters. These parameters include orientations, pixels_per_cell and cells_per_block.

The number of orientations is specified as an integer, and represents the number of orientation bins that the gradient information will be split up into in the histogram. Typical values are between 6 and 12 bins.

The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. This paramater is passed as a 2-tuple so you could have different cell sizes in x and y, but cells are commonly chosen to be square.

The cells_per_block parameter is also passed as a 2-tuple, and specifies the local area over which the histogram counts in a given cell will be normalized. Block normalization is not necessarily required, but generally leads to a more robust feature set.

```
from skimage.feature import hog
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
``` 

Let's say we are computing HOG features for an image that is 64×64 pixels. If you set pixels_per_cell=(8, 8) and cells_per_block=(2, 2) and orientations=9. The HOG features for all cells in each block are computed at each block position and the block steps across and down through the image cell by cell.

So, the actual number of features in your final feature vector will be the total number of block positions multiplied by the number of cells per block, times the number of orientations, or in the case shown above: 7×7×2×2×9=1764.

### Normalization and combining features:
Variety of features helps us in roboust detection system hence we normalize and combine features.

Normalize using sklearn's StandardScaler():
  `
  import numpy as np
  feature_list = [feature_vec1, feature_vec2, ...]
  
  # Create an array stack, NOTE: StandardScaler() expects np.float64
  X = np.vstack(feature_list).astype(np.float64)

  from sklearn.preprocessing import StandardScaler
  # Fit a per-column scaler
  X_scaler = StandardScaler().fit(X)
  # Apply the scaler to X
  scaled_X = X_scaler.transform(X)
  `
Combine features:
`
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features

`

### Build a classifier
Steps:
1. Extract features (color and hog features) from list of images
2. Combine features
2. Shuffle the input data (provided) - to avoid problems due to ordering
2. Split the data into training and testing set - to avoid overfitting and improve generalization
`
  from sklearn.cross_validation import train_test_split
  rand_state = np.random.randint(0, 100)
  X_train, X_test, y_train, y_test = train_test_split(
  scaled_X, y, test_size=0.2, random_state=rand_state)
`
3. Train a classifier to detect car images from other images using ....
`
from sklearn.svm import LinearSVC
svc = LinearSVC()
# Train the SVC
svc.fit(X_train, y_train)
`
5. Check accuracy
`print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
`

5. 
Predicted output:
`
print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
print('For labels: ', y_test[0:10])

`
I tried with these classifier and finally choose this:

I  tweaked these paramerets:
### TODO: Tweak these parameters and see how the results change.
colorspace = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"


#### Sliding window
1. If accuracy is good then will run this classifier across entire frame sampling small patches to detect presence of car in a grid pattern.
2. From our successful prediction we can get start and stop positions in both x and y 
3. From this list of bounding boxes for the search windows we can draw rectangles using draw draw_boxes() function.

### Multiscape search
We are not sure what's the scale of the image we are searching (for example: cars far away appear smaller and closes ones appear large) hence we will set few minimum, maximum and intermediate scales to search.

### Hog Sub-sampling Window Search
Instead of extracting hog features for every small patch, we will extract hog features once and sub small to get all windows/boxes.


### False postives
As seen below we will get multiple detections for the same car and also a false positive, we should filter these out

We will build a heat map to combine overlapping detections and remove false positives.
- To make a heat-map, we're simply going to add "heat" (+=1) for all pixels within windows where a positive detection is reported by your classifier.
- Due to above, areas of multiple detections get "hot", while transient false positives stay "cool". You can then simply threshold your heatmap to remove false positives.
- By manually checking number of false postivies windows (on test images) we can set a threshold to remove it
- To figure out how many cars we have in each frame and which pixels belong to which cars, we use the label() function from scipy.ndimage.measurements.
`heatmap = threshold(heatmap, 2)
labels = label(heatmap)
print(labels[1], 'cars found')
plt.imshow(labels[0], cmap='gray')
`
- We can take our thresholded and labeled images and put bounding boxes around the labeled regions, so that we get single box instead of multiple detections for the same car 

#### Image pipeline

###  Video pipeline
