# -*- coding: utf-8 -*-

import cv2
import numpy as np

src = cv2.imread('sample2.png')

# Convert image to gray and blur it
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('src_grey')
keep_while_loop = True
bin_thresh1 = 100

while keep_while_loop:
    kret = cv2.waitKey(1000) & 0xFF  # 1000ミリ秒の間キー入力を受け付ける
    if kret == ord('q'):  # qを押したら実行
        keep_while_loop = False  # while loopを脱出
    elif kret == ord('o'):
        bin_thresh1 = bin_thresh1+5  # 二値化の閾値をあげる
        print('bin_thresh1=', bin_thresh1)
    elif kret == ord('p'):
        bin_thresh1 = bin_thresh1-5  # 二値化の閾値をさげる
        print('bin_thresh1=', bin_thresh1)

    cv2.imshow('src_grey', src_gray)  # グレースケール画像を表示

    ret, bin_img = cv2.threshold(src_gray, bin_thresh1, 255,
                                 cv2.THRESH_BINARY)  # 二値化画像生成
    ret, bin_imgRev = cv2.threshold(src_gray, bin_thresh1, 255,
                                    cv2.THRESH_BINARY_INV)  # 反転した二値化画像生成

    cv2.imshow('bin_img', bin_img)  # 二値化画像生成
    cv2.imshow('reverse bin_img', bin_imgRev)  # 反転した二値化画像生成

    contours, hierarchy = cv2.findContours(bin_img,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # 輪郭抽出

    # Find the rotated rectangles
    minRect = [None]*len(contours)  # 四角囲みの配列確保
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)  # 囲んでいる四角を取得，代入

    drawing = np.zeros(
        (bin_img.shape[0], bin_img.shape[1], 3), dtype=np.uint8)  # 輪郭度を描画する配列
    for i, c in enumerate(contours):
        color1 = (0, 255, 255)
        color2 = (255, 128, 0)
        # contour
        cv2.drawContours(drawing, contours, i, color1)  # 輪郭を描画

        # rotated rectangle
        box = cv2.boxPoints(minRect[i])  # 囲んでいる四角の頂点を計算
        box = np.intp(box)  # 整数化
        cv2.drawContours(drawing, [box], 0, color2)  # 四角を描画

    # Get the moments
    mu = [None]*len(contours)  # モーメントのデータを代入する配列を確保
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])

    for i in range(len(contours)):
        print(' Contour[%d] | Area : %.2f|Length: %.2f'
              % (i, cv2.contourArea(contours[i]), cv2.arcLength(contours[i], True)))  # オブジェクトの面積，周囲長を表示
        print(' Contour[%d] | mu02 = %.2f | mu20: %.2f|mu11: %.2f'
              % (i, mu[i]['mu02'], mu[i]['mu20'], mu[i]['mu11'],))  # モーメントなどを表示
        theta_rad = 0.5 * \
            np.arctan2(2 * mu[i]['mu11'], mu[i]['mu20']-mu[i]['mu02'])
        print(' Contour[%d] | angle= %.2f'
              % (i, np.degrees(theta_rad)))  # 傾きを表示

    cv2.imshow('Contours', drawing)  # 輪郭(黄色)と囲んでいる四角(青色)を表示


cv2.destroyAllWindows()  # 全てのOpenCVウィンドウを閉じる
