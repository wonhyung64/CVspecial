# Image Filtering

## Original Images
![Original](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/original.png "Original imgs")

## Mean Filtering
![Mean Filtering](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/mean_filter.png "mean filterd imgs")
- Mean filtering with $(7, 7)$ kernal.
## Sharpeness Filtering
![Sharpeness Filtering](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/sharpness_filter.png "shaprned imgs")
- Shapeness Filtering with $\alpha = 10$.

## Gray Scale 
![Gray Scaling](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/gray_filter.png "gray scale imgs")
- Convert RGB image to gray scale image.

## Thresholding Filtering
![Thresholding Filtering](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/thresholding_filter.png "thresholding filterd imgs")
- Thresholding Filtering gray scale image with $threshold = 127.5$.

## Gaussian Filtering
- Gaussian Filtering with $(7, 7)$ kernal size and $\sigma \in \{1.0 ,5.0, 10.0, 30.0\}$

![Gaussian Filtering1](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/gaussian_filter_img1.png "gaussian1")

![Gaussian Filtering2](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/gaussian_filter_img2.png "gaussian2")

![Gaussian Filtering3](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/gaussian_filter_img3.png "gaussian3")

![Gaussian Filtering4](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/gaussian_filter_img4.png "gaussian4")
- $\sigma$가 커질수록 이미지가 더 blur 해지는것을 확인 할 수 있다.

## Hybrid Image
- Blending low and high frequency data.
- Generate Low frequency image by gaussian fitering with $(7,7)$ kernel size, $\sigma = 30$.
- Generate figh frequency image by high-pass filtering with gaussian filter, $(7,7)$ kernel size, $\sigma = 30$. 

![Hybrid Img1](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/hybrid_img1_img2.png "hybrid1")

![Hybrid Img2](https://github.com/wonhyung64/CVspecial/blob/main/image_filtering/src/ex/results/hybrid_img3_img4.png "hybrid2")
- 이미지를 가까이에서 봤을 때는 high frequency 부분이 조금 보이지만 멀리서 봤을 때는 low frequency 부분이 눈에 띄는것을 확인 할 수 있다.
