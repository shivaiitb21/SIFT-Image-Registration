           "IMPLEMENT SIFT BASED FEATURE MATCHING FOR REGISTERING TWO IMAGES"
					By,
			   	 Shivanand Nalgire,
			Indian Institute of Technology Bombay 
			   

....................................REQUIREMENTS.......................................

1. Python 3
2. Numpy
3. Open CV 3.4.1 

...............................PROJECT FOLDER DETAILS..................................

The main project folder contains of 3 folders named:
1.	Code - It contains the 'Code' of project and the 'images - data used'
2.	Output - It has 4 Output folders for 4 different test images.
	Each folder contains 4 images and one text file:
   a.	Original image with features extracted
   b.	Test image with feature extracted
   c.	Image showing matched features
   d.	Output image which is also the Registered Image
   e.	Text file - Containing the data about SIFT features
3.	PPT - This folder contains the Presentation of the Project with all the details

...................................CODE DETAILS........................................

SIFT based Feature Matching for Image Registration has been implemented in 5 
steps as follows:
1. Step-1: 
   a.	Reading the images, original image and test image 

2. Step-2: 
   a.	Extraction of SIFT features - Detecting and Computing the Keypoints 
        and Descriptors using Open CV 3.4.1 Library of python
   b.	Drawing the keypoints on images to visualize the detected keypoints

3. Step-3:
   a.	Match the extracted features using the descriptors of two images 
        (original and test)
   b.	The Brute Force feature matching is used to find the matches
   c.	Norm 'L2' has been used for feature matching (by calculating the 
        neighbourhood using Euclidian Distance)
   d.	The matches have been sorted based upon the distance - least the 
        distance better the match
   e.	For feature matching, only top 90% of matches were used 
   f.	Drawing the matches on Images to visualize the feature matching

4. Step-4:
   a.	Finding the homography matrix for image alignment using the 
        Random sample consensus (RANSAC) method to remove the 
        influence of outliers in features matching

5. Step-5:
   a.	Image warping is done using the Homography matrix generated in 
        above step to correct the distortions in image
   b.	Using the image warping, the output image - which is 'the registered 
        image using feature matching based upon SIFT algorithm' is obtained.



