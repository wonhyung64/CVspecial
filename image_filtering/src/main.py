#%%
from utils import (
    load_data,
    preprocessing,
    array_to_img,
    draw_outputs,
    gaussian_filter,
    mean_filter,
    sharpen_filter,
    gray_filter,
    thresholding_filter,
    highpass_filter,
)

#%%
if __name__ == "__main__":
    imgs = load_data()
    img0, img1, img2, img3 = imgs = preprocessing(imgs)

    imgs_org = [array_to_img(img) for img in imgs]
    imgs_mean = [array_to_img(mean_filter(img, 7)) for img in imgs]
    imgs_sharp = [array_to_img(sharpen_filter(img, factor=10)) for img in imgs]
    imgs_gray = [array_to_img(gray_filter(img)) for img in imgs]
    imgs_thresh = [array_to_img(thresholding_filter(gray_filter(img))) for img in imgs]

    gaussian_sigma = [1.0, 5.0, 10.0, 30.0]
    img0_gaussian = [
        array_to_img(gaussian_filter(img0, 7, sigma)) for sigma in gaussian_sigma
    ]
    img1_gaussian = [
        array_to_img(gaussian_filter(img1, 7, sigma)) for sigma in gaussian_sigma
    ]
    img2_gaussian = [
        array_to_img(gaussian_filter(img2, 7, sigma)) for sigma in gaussian_sigma
    ]
    img3_gaussian = [
        array_to_img(gaussian_filter(img3, 7, sigma)) for sigma in gaussian_sigma
    ]

    detail_ext = [highpass_filter(img, gaussian_filter(img, 7, 30.0)) for img in imgs]
    hybrid1 = [
        array_to_img(detail_ext[0]),
        array_to_img(gaussian_filter(img1, 7, 30) + detail_ext[0]),
        array_to_img(detail_ext[1]),
        array_to_img(gaussian_filter(img0, 7, 30) + detail_ext[1]),
    ]
    hybrid2 = [
        array_to_img(detail_ext[2]),
        array_to_img(gaussian_filter(img3, 7, 30) + detail_ext[2]),
        array_to_img(detail_ext[3]),
        array_to_img(gaussian_filter(img2, 7, 30) + detail_ext[3]),
    ]

    result = [
        imgs_org,
        imgs_mean,
        imgs_sharp,
        imgs_gray,
        imgs_thresh,
        img0_gaussian,
        img1_gaussian,
        hybrid1,
        img2_gaussian,
        img3_gaussian,
        hybrid2,
    ]

    for idx in range(len(result)):
        if idx == 3 or idx == 4:
            cmap = "gray"
        else:
            cmap = None
        draw_outputs(result[idx], cmap)
