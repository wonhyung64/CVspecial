#%%
from utils import (
    load_data,
    array_to_img,
    gray_filter,
    thresholding_filter,
    gaussian_filter,
    laplassian_filter,
    canny_edge,
    sobel_edge,
)

#%%
imgs = load_data()
img = imgs[0]



img_gray = gray_filter(img)
array_to_img(img_gray)


img_blur = gaussian_filter(img_gray, 7, 0.)
array_to_img(img_blur)

img_blur_log = laplassian_filter(img_gray, 5, 1)
array_to_img(img_blur_log)
#%%
    
sobel_x = sobel_edge(img_blur, "x", 5)
sobel_y = sobel_edge(img_blur, "y", 5)
sobel_xy = sobel_edge(img_blur, "xy", 5)

array_to_img(sobel_x)
array_to_img(sobel_y)
array_to_img(sobel_xy)
array_to_img(sobel_x + sobel_y)


threshold = 127.5
sobel = thresholding_filter(sobel_x + sobel_y, threshold)
array_to_img(sobel)


gaussian_canny = canny_edge(img_blur, threshold1=100, threshold2=200.)
log_canny = canny_edge(img_blur_log, threshold1=100, threshold2=200)
array_to_img(gaussian_canny)
array_to_img(log_canny)


# %%
