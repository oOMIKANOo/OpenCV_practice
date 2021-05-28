# -*- coding: utf-8 -*-

import serial
from serial.tools import list_ports
import sys
import cv2
import numpy as np
import csv
import time
import math


length_of_memory = int(100)


def log_input(data_array, index_num, data1,
              data2=0.0, data3=0.0, data4=0.0,
              data5=0.0, data6=0.0, data7=0.0,
              data8=0.0, data9=0.0, data10=0.0):
    data_array[index_num][0] = float(data1)
    data_array[index_num][1] = float(data2)
    data_array[index_num][2] = float(data3)
    data_array[index_num][3] = float(data4)
    data_array[index_num][4] = float(data5)
    data_array[index_num][5] = float(data6)
    data_array[index_num][6] = float(data7)
    data_array[index_num][7] = float(data8)
    data_array[index_num][8] = float(data9)
    data_array[index_num][9] = float(data10)

# pt0-> pt1およびpt0-> pt2からの
# ベクトル間の角度の余弦(コサイン)を算出


def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0, 0] - pt0[0, 0])
    dy1 = float(pt1[0, 1] - pt0[0, 1])
    dx2 = float(pt2[0, 0] - pt0[0, 0])
    dy2 = float(pt2[0, 1] - pt0[0, 1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2))
    return (dx1*dx2 + dy1*dy2) / v


# 画像上の四角形を検出し、不要なものは削除
def findSquares(contours, cond_area=50):

    for i, cnt in enumerate(contours):
        # 輪郭の周囲に比例する精度で輪郭を近似する
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, arclen*0.02, True)

        # 四角形の輪郭は、近似後に4つの頂点があります。
        # 比較的広い領域が凸状になります。

        # 凸性の確認
        area = abs(cv2.contourArea(approx))
        if approx.shape[0] == 4 and area > cond_area and cv2.isContourConvex(approx):
            maxCosine = 0

            for j in range(2, 5):
                # 辺間の角度の最大コサインを算出
                cosine = abs(angle(approx[j % 4], approx[j-2], approx[j-1]))
                maxCosine = max(maxCosine, cosine)

            # すべての角度の余弦定理が小さい場合
            # （すべての角度は約90度です）次に、quandrangeを書き込みます
            # 結果のシーケンスへの頂点
            if maxCosine > 0.3:
                # 四角判定
                del contours[i]
        else:
            del contours[i]


def main():  # メイン関数を定義する．
    global length_of_memory
    log_memory = np.zeros((length_of_memory, 10), dtype=float)
    log_index = 0
    log_exec = False

    # シリアル通信状態
    serial_switch = 0

    # 2値化の閾値
    Threshold_change = 120

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # カメラの初期化
    if capture.isOpened() != True:
        print('camera device opening error')
        sys.exit()

    # ports = list_ports.comports()  # COMポートがひとつだけある場合はこれで見つかる．
    #Ser = serial.Serial(ports[0].device, 9600)

    Ser = serial.Serial('COM4', 9600)  # COMポートが2個以上ある場合はこちらで指定
    if Ser.isOpen() != True:
        cv2.destroyAllWindows()  # 全てのOpenCVウィンドウを閉じる
        sys.exit()  # 終了

    start_time = time.time()

    keep_while_loop = True  # while loopを継続する
    while keep_while_loop:
        ret, frame = capture.read()  # カメラからデータを読み込むretには結果(True or False)が，frameには画像データがはいる
        windowsize = (800, 600)
        frame = cv2.resize(frame, windowsize)  # ウィンドウサイズを小さくする
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayによる画像情報に変換
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # frameに戻す

        #########################################
        # ここに処理を追加する．
        #########################################

        bin_image = cv2.inRange(frame, np.array(
            [Threshold_change, 0, 0]), np.array([255, 255, 255]))

        # 適応的二値化処理
        # frame01 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # bin_image = cv2.adaptiveThreshold(frame01, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # オープニング・クロージングによるノイズ除去
        element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
        bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, element8)
        bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, element8)

        cv2.imshow('Gray', frame)  # gray画像を表示
        cv2.imshow('Two', bin_image)

        contours, hierarchy = cv2.findContours(
            bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭抽出

        # 面積順に並び替え
        contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)

        # 面積の大きいものだけ残す
        del contours[9:]

        del contours[0]

        # 周囲長の短い順に並び替え
        contours.sort(key=lambda y: cv2.arcLength(y, True), reverse=True)

        # 要素を二つだけ残す
        del contours[0:1]

        findSquares(contours)

        # Find the rotated rectangles
        minRect = [None]*len(contours)  # 四角囲みの配列確保
        for i, c in enumerate(contours):
            minRect[i] = cv2.minAreaRect(c)  # 囲んでいる四角を取得，代入

        drawing = np.zeros(
            (bin_image.shape[0], bin_image.shape[1], 3), dtype=np.uint8)  # 輪郭度を描画する配列
        for i, c in enumerate(contours):
            color1 = (0, 255, 255)
            color2 = (255, 128, 0)
            # contour
            cv2.drawContours(drawing, contours, i, color1)  # 輪郭を描画

            # rotated rectangle
            box = cv2.boxPoints(minRect[i])  # 囲んでいる四角の頂点を計算
            box = np.intp(box)  # 整数化
            cv2.drawContours(drawing, [box], 0, color2)  # 四角を描画

        # 四角形の座標保管
        Coordinate = [[0, 0], [0, 0], [0, 0], [
            0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

        # 角度計算用の四角形の座標
        # 0にx座標,1にy座標
        up_shikaku_small = [0, 0]
        up_shikaku_big = [0, 0]
        down_shikaku_small = [0, 0]
        down_shikaku_big = [0, 0]

        # Get the moments
        mu = [None]*len(contours)  # モーメントのデータを代入する配列を確保
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])

        for i in range(len(contours)):

            Coordinate[i][1] = mu[i]["m10"]/mu[i]["m00"]  # 重心のx座標
            Coordinate[i][0] = mu[i]["m01"]/mu[i]["m00"]  # 重心のy座標

        #　y座標の大きい順にソート
        Coordinate.reverse

        # 重心のy座標の大きさで角度計算に使えるように並び替え・整理
        down_shikaku_big[0] = Coordinate[0][1]
        down_shikaku_big[1] = Coordinate[0][0]
        down_shikaku_small[0] = Coordinate[1][1]
        down_shikaku_small[1] = Coordinate[1][0]
        up_shikaku_big[0] = Coordinate[2][1]
        up_shikaku_big[1] = Coordinate[2][0]
        up_shikaku_small[0] = Coordinate[3][1]
        up_shikaku_small[1] = Coordinate[3][0]

        # シータ1の角度計算
        dx = -(down_shikaku_small[1]-down_shikaku_big[1])
        dy = -(down_shikaku_small[0]-down_shikaku_big[0])
        angle01 = np.arctan2(dy, dx)*180.0/3.14159
        angle01 = np.round(angle01, 1)

        # シータ2の角度計算
        dx = -(up_shikaku_small[1]-up_shikaku_big[1])
        dy = -(up_shikaku_small[0]-up_shikaku_big[0])
        angle02_provisional = np.arctan2(dy, dx)*180.0/3.14159
        angle02_provisional = np.round(angle02_provisional, 1)
        angle02 = angle02_provisional-angle01
        angle02 = np.round(angle02, 1)

        print('angle01=', angle01, '[deg]')
        print('angle02=', angle02, '[deg]')

        x0 = [0.0]*len(contours)  # データを代入する配列を確保
        y0 = [0.0]*len(contours)  # データを代入する配列を確保
        vx = [0.0]*len(contours)  # データを代入する配列を確保
        vy = [0.0]*len(contours)  # データを代入する配列を確保

        for i in range(len(contours)):
            [vx[i], vy[i], x0[i], y0[i]] = cv2.fitLine(
                contours[i], cv2.DIST_L2, 0, 0.01, 0.01)  # 直線の当てはめ

            line_len = 200.0
            x1 = int(x0[i]-line_len*vx[i])
            y1 = int(y0[i]-line_len*vy[i])
            x2 = int(x0[i]+line_len*vx[i])
            y2 = int(y0[i]+line_len*vy[i])
            cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Contours', drawing)  # 輪郭(黄色)と囲んでいる四角(青色)を表示

        # シリアル送信
        if(serial_switch == 1):
            if angle01 > 47:  # 2度分余裕をもたせる
                send_char = '3\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信
            elif angle01 < 43:
                send_char = '1\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信
            else:
                send_char = '2\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信

            if angle02 > 47:
                send_char = '6\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信
            elif angle02 < 43:
                send_char = '4\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信
            else:
                send_char = '5\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信

        if log_exec == True:
            current = time.time() - start_time
            log_input(log_memory, log_index, current, data2=current *
                      current, data3=angle01, data4=angle02)

            log_index = log_index + 1
            if log_index == length_of_memory:
                log_exec = False
                print('log finished!')

        kret = cv2.waitKey(1) & 0xFF  # 1ミリ秒の間キー入力を受け付ける
        if kret == ord('q'):  # qを押したら実行
            keep_while_loop = False  # while loopを脱出
        elif kret == ord('r'):  # rを押したら実行
            log_index = 0
            log_exec = True  # logスタート
            print('log start!')
        elif kret == ord('o'):
            Threshold_change = Threshold_change+5  # 閾値変更
        elif kret == ord('p'):
            Threshold_change = Threshold_change-5  # 閾値変更
        elif kret == ord('s'):  # シリアル通信開始
            serial_switch = 1
        elif kret == ord('x'):  # シリアル通信停止
            serial_switch = 0
            send_char = '2\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信
            send_char = '5\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信
        # 設定用手動操作
        elif kret == ord('1'):
            send_char = '1\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信
        elif kret == ord('2'):
            send_char = '2\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信
        elif kret == ord('3'):
            send_char = '3\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信
        elif kret == ord('4'):
            send_char = '4\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信
        elif kret == ord('5'):
            send_char = '5\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信
        elif kret == ord('6'):
            send_char = '6\r\n'  # 改行記号をつけて
            send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
            Ser.write(send_char_encoded)  # シリアル送信

    # while loopを抜けたあと
    send_char = '2\r\n'  # 改行記号をつけて
    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
    Ser.write(send_char_encoded)  # シリアル送信
    send_char = '5\r\n'  # 改行記号をつけて
    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
    Ser.write(send_char_encoded)  # シリアル送信
    Ser.close()  # シリアルを閉じる
    capture.release()  # カメラデバイスを解放
    cv2.destroyAllWindows()  # 全てのOpenCVウィンドウを閉じる

    if log_index != 0:
        # ログデータをCSV形式で保存する．
        f = open('data.csv', 'w')
        csvWriter = csv.writer(f, lineterminator='\n')  # csvWriteの実装
        csvWriter.writerows(log_memory)  # CSVデータとして保存
        f.close()


# Pythonのメイン関数
if __name__ == "__main__":
    main()
