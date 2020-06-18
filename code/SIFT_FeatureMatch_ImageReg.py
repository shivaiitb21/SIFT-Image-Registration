import cv2
import numpy as np

def sift_image_registration(x, y):

      ####....................STEP-1: Read images....................####
      img1 = cv2.imread(x)
      img1 = cv2.resize(img1, (350, 512), interpolation=cv2.INTER_AREA)
      img2 = cv2.imread(y)
      height, width, ch = img2.shape

      ####..............STEP-2: SIFT Feature Extraction..............####
      # SIFT feature extractor
      sift = cv2.xfeatures2d_SIFT.create()

      # Extract keypoints and descriptors in image
      kp1, des1 = sift.detectAndCompute(img1, None)
      kp2, des2 = sift.detectAndCompute(img2, None)

      # Drawing keypoints --> Extracted on the images
      img1 = cv2.drawKeypoints(img1, kp1, None)
      img2 = cv2.drawKeypoints(img2, kp2, None)

      ####...........STEP-3: Brute-Force Feature Matching...........####
      # Brute force matcher
      # With 'L2' Norm --> Eucladian distance
      bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

      # Match the two sets of descriptors.
      matches = bf.match(des1, des2)

      # Sorted matches on the basis of their Eucladian distance.
      matches = sorted(matches, key=lambda x: x.distance)

      # Taking 90% of forward matches
      matches = matches[:int(len(matches) * 90)]
      no_of_matches = len(matches)

      # Draw the matched keypoints on two images
      matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)

      ####....STEP-4: Feature Based IMAGE WARPING AND IMAGE ALIGNMENT....####
      # Define empty matrices of shape no_of_matches * 2.
      p1 = np.zeros((no_of_matches, 2))
      p2 = np.zeros((no_of_matches, 2))

      for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

      # Find the homography matrix --> Using Random sample consensus (RANSAC)
      homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

      ####...........STEP-5: Image Registration..........####
      # Use homography matrix to transform the
      # colored image wrt the reference image.
      transformed_img = cv2.warpPerspective(img1, homography, (width, height))

      # Save all the images and output image (Registered image)
      # Image to be registered
      cv2.imwrite('1_Original_image.jpg', img1)
      cv2.imwrite('2_Test_image.jpg', img2)
      cv2.imwrite('3_Feature_matching.jpg', matching_result)
      # Registred image as output (from Test_image)
      cv2.imwrite('4_Output.jpg', transformed_img)

      # To show the images
      cv2.imshow('1_Original_image.jpg', img1)
      cv2.imshow('2_Test_image.jpg', img2)
      cv2.imshow('3_Feature_matching.jpg', matching_result)
      cv2.imshow('4_Output.jpg', transformed_img)

      cv2.waitKey(0)
      cv2.destroyAllWindows()

      # Taking statistical output of SIFT Features
      import sys
      orig_stdout = sys.stdout
      f = open('Output.txt', 'w')
      sys.stdout = f
      print("STATISTICAL DATA OF IMPLEMENTATION OF 'SIFT' BASED FEATURES")
      print()
      print("1) No of descriptors / keypoints of Original image: ", len(des1))
      print("2) No of descriptors / keypoints of Test image: ", len(des2))
      print("3) No of matches in Original image and Test Image: ", no_of_matches)
      sys.stdout = orig_stdout
      f.close()

# Testing the function on Images
sift_image_registration("book.jpg", "test1.jpeg")
sift_image_registration("book.jpg", "test2.jpeg")
sift_image_registration("book.jpg", "test3.jpeg")
sift_image_registration("book.jpg", "test4.jpeg")


