# Edge Detection

## Images
![Original](https://github.com/wonhyung64/CVspecial/blob/main/edge_detection/src/ex/results/original.png "Original imgs")
- (1) Original image
- (2) Gray scale image
- (3) Gaussian filtered image with (7,7) kernel and 1 sigma
- (4) Laplassian filtered image with (5,5) kernel and 1 sigma

## Sobel Edge Detection
![Sobel Edge Detection](https://github.com/wonhyung64/CVspecial/blob/main/edge_detection/src/ex/results/sobel.png "Sobel edge detection")
- sobel edge detection on gaussian filtered image with (7,7) kernel and 1 sigma
- (1) x derivatives edge with (5,5) kernel
- (2) y derivatives edge with (5,5) kernel
- (3) x and y derivatives edge with (5,5) kernel
- (4) x derivatives + y derivatives edge with (5,5) kernel

## Sobel Edge Detection with thresholding filter
![Sobel Edge Detection w thresholding](https://github.com/wonhyung64/CVspecial/blob/main/edge_detection/src/ex/results/sobel_thresholding.png "sobel with thresholding filter")
- thresholding filter on xy derivatives sobel edge detected image
- (1) 20 threshold
- (2) 30 threshold
- (3) 40 threshold
- (4) 50 threshold

## Canny Edge Detection
![Canny Edge Detection](https://github.com/wonhyung64/CVspecial/blob/main/edge_detection/src/ex/results/gaussian_canny.png "canny edge detection")
- canny edge detection on gaussian filtered image with (7,7) kernel and 1 sigma
- low level threshold is fixed with 10
- (1) 70 for high level threshold
- (2) 90 for high level threshold
- (3) 110 for high level threshold
- (4) 130 for high level threshold

## Edge Detection with Laplassian Filter
![Laplassian Filter](https://github.com/wonhyung64/CVspecial/blob/main/edge_detection/src/ex/results/laplassina_thresholding.png "laplassian filter and thresholding")
-  thresholding filter on laplassian filtered image with (5,5) kernel and 1 sigma
- (1) 100 threshold
- (2) 150 threshold
- (3) 200 threshold
- (4) 250 threshold
