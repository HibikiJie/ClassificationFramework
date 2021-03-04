from random import randint
import random
import numpy
import cv2
import math


def gauss_blur(image, max_level):
    """
    高斯模糊
    :param image: 图片
    :param max_level: 最大模糊等级
    :return:
    """
    level = randint(0, max_level)
    return cv2.blur(image, (level * 2 + 1, level * 2 + 1))


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = numpy.zeros(image.shape, numpy.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = random.randint(0, 255)
            elif rdn > thres:
                output[i][j] = random.randint(0, 255)
            else:
                output[i][j] = image[i][j]
    return output


def randomly_adjust_brightness(image, lightness, saturation):
    # 颜色空间转换 BGR转为HLS
    image = image.astype(numpy.float32) / 255.0
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # 1.调整亮度（线性变换)
    hlsImg[:, :, 1] = (1.0 + lightness / float(100)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # 饱和度
    hlsImg[:, :, 2] = (1.0 + saturation / float(100)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(numpy.uint8)
    return lsImg


def random_cutting(image, size):
    """
    随机裁剪
    :param image: 图片
    :param size: 裁剪尺寸
    :return:
    """
    h, w, _ = image.shape
    a = random.randint(0, size)
    b = random.randint(0, size)
    image = image[a:h - a, b:w - b]
    return image


def reset_image(image, image_size, is_random_pation):
    h1, w1, _ = image.shape
    max_len = max(h1, w1)
    fx = image_size / max_len
    fy = image_size / max_len
    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    h2, w2, _ = image.shape
    background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
    if is_random_pation:
        s_h = random.randint(0, image_size - h2)
        s_w = random.randint(0, image_size - w2)
    else:
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
    background[s_h:s_h + h2, s_w:s_w + w2] = image
    return background


def transform_matrix(pts, t_pts):
    return cv2.getPerspectiveTransform(numpy.float32(pts[:2, :].T), numpy.float32(t_pts[:2, :].T))


def points_matrix(pts):
    return numpy.matrix(numpy.concatenate((pts, numpy.ones((1, pts.shape[1]))), 0))


def rect_matrix(tlx, tly, brx, bry):
    return numpy.matrix([
        [tlx, brx, brx, tlx],
        [tly, tly, bry, bry],
        [1.0, 1.0, 1.0, 1.0]
    ])


def rotate_matrix(width, height, angles=numpy.zeros(3), zcop=1000.0, dpp=1000.0):
    rads = numpy.deg2rad(angles)
    rx = numpy.matrix([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(rads[0]), math.sin(rads[0])],
        [0.0, -math.sin(rads[0]), math.cos(rads[0])]
    ])
    ry = numpy.matrix([
        [math.cos(rads[1]), 0.0, -math.sin(rads[1])],
        [0.0, 1.0, 0.0],
        [math.sin(rads[1]), 0.0, math.cos(rads[1])]
    ])
    rz = numpy.matrix([
        [math.cos(rads[2]), math.sin(rads[2]), 0.0],
        [-math.sin(rads[2]), math.cos(rads[2]), 0.0],
        [0.0, 0.0, 1.0]
    ])
    r = rx * ry * rz
    hxy = numpy.matrix([
        [0.0, 0.0, width, width],
        [0.0, height, 0.0, height],
        [1.0, 1.0, 1.0, 1.0]
    ])
    xyz = numpy.matrix([
        [0.0, 0.0, width, width],
        [0.0, height, 0.0, height],
        [0.0, 0.0, 0.0, 0.0]
    ])
    half = numpy.matrix([[width], [height], [0.0]]) / 2.0
    xyz = r * (xyz - half) - numpy.matrix([[0.0], [0.0], [zcop]])
    xyz = numpy.concatenate((xyz, numpy.ones((1, 4))), 0)
    p = numpy.matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0 / dpp, 0.0]
    ])
    t_hxy = p * xyz
    t_hxy = t_hxy / t_hxy[2, :] + half
    return transform_matrix(hxy, t_hxy)


def project(img, pts, trans, dims):
    t_img = cv2.warpPerspective(img, trans, (dims, dims))
    t_pts = numpy.matmul(trans, points_matrix(pts))
    t_pts = t_pts / t_pts[2]
    return t_img, t_pts[:2]


def reconstruct_image(image, points, out_size=(112, 112)):
    """
    根据点位，抠出图片，四点
    :param image: 图片
    :param points: 点位
    :param out_size: 输出图片尺寸
    :return:
    """
    wh = numpy.array([[image.shape[1]], [image.shape[0]]])
    images = []
    for point in points:
        point = points_matrix(point * wh)
        t_pts = rect_matrix(0, 0, out_size[0], out_size[1])
        m = transform_matrix(point, t_pts)
        plate = cv2.warpPerspective(image, m, out_size)
        images.append(plate)
    return images


def augment_sample(image, dims=416):
    """图片的空间增广"""
    points = [val + random.uniform(-0.1, 0.1) for val in [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]]
    points = numpy.array(points).reshape((2, 4))
    points = points * numpy.array([[image.shape[1]], [image.shape[0]]])
    wh_ratio = random.uniform(2.0, 4.0)
    width = random.uniform(dims * 0.2, dims * 1.0)
    height = width / wh_ratio
    dx = random.uniform(0.0, dims - width)
    dy = random.uniform(0.0, dims - height)
    crop = transform_matrix(
        points_matrix(points),
        rect_matrix(dx, dy, dx + width, dy + height)
    )
    max_angles = numpy.array([80.0, 80.0, 45.0])
    angles = numpy.random.rand(3) * max_angles
    if angles.sum() > 120:
        angles = (angles / angles.sum()) * (max_angles / max_angles.sum())
    rotate = rotate_matrix(dims, dims, angles)
    image, points = project(image, points, numpy.matmul(rotate, crop), dims)
    points = points / dims
    return image, numpy.asarray(points).reshape((-1,)).tolist()


def square_picture(image, image_size):
    """
    任意图片正方形中心化
    :param image: 图片
    :param image_size: 输出图片的尺寸
    :return: 输出图片
    """
    h1, w1, _ = image.shape
    max_len = max(h1, w1)
    fx = image_size / max_len
    fy = image_size / max_len
    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    h2, w2, _ = image.shape
    background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
    s_h = image_size // 2 - h2 // 2
    s_w = image_size // 2 - w2 // 2
    background[s_h:s_h + h2, s_w:s_w + w2] = image
    return background
