#%%
from utils import (
    load_data,
    array_to_img,
    draw_outputs,
    gray_filter,
    thresholding_filter,
    gaussian_filter,
    laplassian_filter,
    canny_edge,
    sobel_edge,
)

#%%
if __name__ == "__main__":
    imgs = load_data()
    img = imgs[0]

    img_gray = gray_filter(img)
    img_blur = gaussian_filter(img_gray, 7, 0.)
    img_blur_log = laplassian_filter(img_gray, 5, 1)

    sobel_x = sobel_edge(img_blur, "x", 5)
    sobel_y = sobel_edge(img_blur, "y", 5)
    sobel_xy = sobel_edge(img_blur, "xy", 5)

    original = [
        array_to_img(img),
        array_to_img(img_gray),
        array_to_img(img_blur),
        array_to_img(img_blur_log),
    ]

    sobel = [
        array_to_img(sobel_x),
        array_to_img(sobel_y),
        array_to_img(sobel_xy),
        array_to_img(sobel_x + sobel_y),
    ]

    sobel_thresholding = [
        array_to_img(thresholding_filter(sobel_xy, 20)), 
        array_to_img(thresholding_filter(sobel_xy, 30)),
        array_to_img(thresholding_filter(sobel_xy, 40)),
        array_to_img(thresholding_filter(sobel_xy, 50)),
    ]

    canny_gaussian = [
        array_to_img(canny_edge(img_blur, threshold1=10, threshold2=70)), 
        array_to_img(canny_edge(img_blur, threshold1=10, threshold2=90)), 
        array_to_img(canny_edge(img_blur, threshold1=10, threshold2=110)),
        array_to_img(canny_edge(img_blur, threshold1=10, threshold2=130)),
    ]

    canny_laplassian = [
        array_to_img(thresholding_filter(img_blur_log, 100)),
        array_to_img(thresholding_filter(img_blur_log, 150)),
        array_to_img(thresholding_filter(img_blur_log, 200)),
        array_to_img(thresholding_filter(img_blur_log, 250)),
    ]

    result = [
        original,
        sobel,
        sobel_thresholding,
        canny_gaussian,
        canny_laplassian,
    ]

    for idx in range(len(result)):
        if idx == 3 or idx == 4:
            cmap = "gray"
        else:
            cmap = None
        draw_outputs(result[idx], cmap)