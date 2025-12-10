import cv2
import numpy as np
import os
from tqdm import *
import json
from argparse import ArgumentParser


def get_rgb_hist(img, bsize=16):
    B = img[..., 0] // (256 // bsize)
    G = img[..., 1] // (256 // bsize)
    R = img[..., 2] // (256 // bsize)
    index = B * bsize * bsize + G * bsize + R
    rgbhist = cv2.calcHist([index], [0], None, [
                           bsize * bsize * bsize], [0, bsize * bsize * bsize])
    return rgbhist


def get_rgb_hist_list(imgs, bsize=16):
    batchsize = len(imgs)
    imgbatch = imgs
    accumulated_hist = np.zeros((bsize * bsize * bsize, 1), dtype=np.float32)
    for img in imgbatch:
        rgbhist = get_rgb_hist(img, bsize)
        accumulated_hist += rgbhist
    rgbhist = accumulated_hist / batchsize
    return rgbhist


def getHsvHist(imgHSV):
    '''
    opencv hsv 范围:
    h(0,180)
    s(0,255)
    v(0,255)
    '''

    height, width, _ = imgHSV.shape
    H = np.zeros((height, width), dtype=np.uint8)
    S = np.zeros((height, width), dtype=np.uint8)
    V = np.zeros((height, width), dtype=np.uint8)

    h = imgHSV[..., 0]
    s = imgHSV[..., 1]
    v = imgHSV[..., 2]

    h = 2*h
    H[(h > 315) | (h <= 20)] = 0
    H[(h > 20) & (h <= 40)] = 1
    H[(h > 40) & (h <= 75)] = 2
    H[(h > 75) & (h <= 155)] = 3
    H[(h > 155) & (h <= 190)] = 4
    H[(h > 190) & (h <= 270)] = 5
    H[(h > 270) & (h <= 295)] = 6
    H[(h > 295) & (h <= 315)] = 7

    '''
    255*0.2 = 51
    255*0.7 = 178
    '''
    S[s <= 51] = 0
    S[(s > 51) & (s <= 178)] = 1
    S[s > 178] = 2

    V[v <= 51] = 0
    V[(v > 51) & (v <= 178)] = 1
    V[v > 178] = 2

    g = 9*H + 3*S + V
    hist = cv2.calcHist([g], [0], None, [72], [0, 71])
    return hist


def getHsvHistList(imgs):
    batchsize = len(imgs)
    imgbatch = imgs
    accumulated_hist = np.zeros((72, 1), dtype=np.float32)
    for img in imgbatch:
        hsvhist = getHsvHist(img)
        accumulated_hist += hsvhist
    hsvhist = accumulated_hist / batchsize
    return hsvhist


def likelihood(img1, img2):
    hist1 = getHsvHist(img1)
    hist2 = getHsvHist(img2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def getImgsFromPath(dir, images):
    assert os.path.isdir(dir) or os.path.islink(
        dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def compare_img(img1, img2, use_rgb_hist=True):
    if use_rgb_hist:
        # input: bgr
        ref_hist = get_rgb_hist(img1)
        gen_hist = get_rgb_hist(img2)
        score = cv2.compareHist(ref_hist, gen_hist, cv2.HISTCMP_CORREL)
        return score
    else:
        gen_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        ref_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        score = likelihood(ref_hsv, gen_hsv)
        return score


def test():
    rgb_hist = False
    ref_img = cv2.imread("ref.jpg")
    wRef_img = cv2.imread("wRef.png")
    # wRef_hsv = cv2.cvtColor(wRef_img, cv2.COLOR_BGR2HSV)
    woRef_img = cv2.imread("woRef.png")
    # woRef_img = cv2.cvtColor(woRef_img, cv2.COLOR_BGR2HSV)
    score = compare_img(ref_img, wRef_img, rgb_hist)
    score2 = compare_img(ref_img, woRef_img, rgb_hist)
    print(score, score2)
    exit(0)


def rgb_hist_score_api(gen_img_paths, ref_img_paths):
    # RGB
    tot_score = 0
    ref_imgs = []
    for ref_img_path in (ref_img_paths):
        ref_img = cv2.imread(ref_img_path)
        ref_imgs.append(ref_img)

    ref_hist = get_rgb_hist_list(ref_imgs)
    for gen_img_path in (gen_img_paths):
        gen_img = cv2.imread(gen_img_path)
        gen_hist = get_rgb_hist(gen_img)
        score = cv2.compareHist(ref_hist, gen_hist, cv2.HISTCMP_CORREL)
        tot_score += score
    avg_score = tot_score / len(gen_img_paths)

    return avg_score


def hsv_hist_score_api(gen_img_paths, ref_img_paths):
    # HSV
    tot_score = 0
    ref_imgs = []
    for ref_img_path in (ref_img_paths):
        ref_img = cv2.imread(ref_img_path)
        ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
        ref_imgs.append(ref_hsv)
    ref_hist = getHsvHistList(ref_imgs)
    for gen_img_path in (gen_img_paths):
        gen_img = cv2.imread(gen_img_path)
        gen_hsv = cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV)
        gen_hist = getHsvHist(gen_hsv)
        score = cv2.compareHist(ref_hist, gen_hist, cv2.HISTCMP_CORREL)
        tot_score += score
    avg_score = tot_score / len(gen_img_paths)
    return avg_score


def hist_score_api(gen_img_paths, ref_img_paths):
    # 主要测试接口
    # 输入:
    # gen_img_paths: 生成图片的路径
    # ref_img_paths: 参考图片的路径
    rgb_score = rgb_hist_score_api(gen_img_paths, ref_img_paths)
    hsv_score = hsv_hist_score_api(gen_img_paths, ref_img_paths)
    avg_score = (rgb_score + hsv_score) / 2
    return avg_score


if __name__ == "__main__":
    # test()
    gen_img_path = "/home/ldy/ldy/jtcomp/test"
    ref_img_path = "/home/ldy/ldy/jtcomp/ref_test"
    gen_img_paths = os.listdir(gen_img_path)
    ref_img_paths = os.listdir(ref_img_path)
    gen_img_paths = [os.path.join(gen_img_path, img) for img in gen_img_paths]
    ref_img_paths = [os.path.join(ref_img_path, img) for img in ref_img_paths]
    print(hist_score_api(gen_img_paths, ref_img_paths))
    exit()
    parser = ArgumentParser()
    parser.add_argument('--ref', help='reference picture folder',
                        type=str, default='./train_resized/imgs')
    parser.add_argument('--gen', help='generated picture folder', type=str)
    parser.add_argument(
        '--worefgen', default=None, type=str)
    parser.add_argument('--rgb', help='use rgb histogram', default=True)
    parser.add_argument('--hsv', help='use hsv histogram', default=True)
    args = parser.parse_args()
    # get mapping relationship of filenames between ref and gen
    # f = open('./ref_cos.txt', 'r')
    # ref_dict = {}
    # for line in f:
    #     a = line.strip().split(',')
    #     ref_dict[a[0]] = a[1]
    # ref_dict = {}
    with open("label_to_img.json", "r") as f:
        ref_dict = json.load(f)  # gen to ref
    # get imgs from ref and gen folder
    ref_imgs = []
    getImgsFromPath(args.ref, ref_imgs)
    gen_imgs = []
    getImgsFromPath(args.gen, gen_imgs)  # 命名与label一致
    # compare them
    tot_score = 0
    if args.rgb:
        for gen_img_path in tqdm(gen_imgs):
            gen_img = cv2.imread(gen_img_path)
            # label_file = gen_img_path.split("/")[-1]
            label_file = os.path.basename(gen_img_path).replace(".jpg", ".png")
            img_file = ref_dict[label_file]
            # ref_img_path = "/".join([args.ref, img_file])
            ref_img_path = os.path.join(args.ref, img_file)

            # print(f'(1) {gen_img_path} \n (2) {ref_img_path} ')
            ref_img = cv2.imread(ref_img_path)

            ref_hist = get_rgb_hist(ref_img)
            gen_hist = get_rgb_hist(gen_img)
            score = cv2.compareHist(ref_hist, gen_hist, cv2.HISTCMP_CORREL)
            tot_score += score
    if args.hsv:  # hsv
        for gen_img_path in tqdm(gen_imgs):
            gen_img = cv2.imread(gen_img_path)
            label_file = os.path.basename(gen_img_path).replace(".jpg", ".png")
            img_file = ref_dict[label_file]
            ref_img_path = "/".join([args.ref, img_file])
            ref_img = cv2.imread(ref_img_path)
            gen_hsv = cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV)
            ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)

            score = likelihood(ref_hsv, gen_hsv)
            tot_score += score
            # print(f'(1) {gen_img_path} \n (2) {ref_img_path} \n score=', score)
            # exit(0)
    if args.rgb and args.hsv:
        avg_score = tot_score / (len(gen_imgs) * 2)
    else:
        avg_score = tot_score / len(gen_imgs)
    print('avg_score:', avg_score)

    with open(f"{args.gen.replace('upload', 'results')}/metrics.txt", 'a+') as f:
        print('Style_similarity: ', avg_score)
        f.write(str(avg_score) + "\n")
