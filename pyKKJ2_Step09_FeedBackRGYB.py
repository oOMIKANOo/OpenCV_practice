# -*- coding: utf-8 -*-

import serial
from serial.tools import list_ports
import sys
import cv2
import numpy as np
import time
import math
import csv

minimum_radius = int(10)
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


def getCircleDualThreshould(hsv_frame, lower_color1, upper_color1, lower_color2, upper_color2):
    global minimum_radius

    # 指定した色範囲のみを抽出する
    bin_image1 = cv2.inRange(hsv_frame, lower_color1, upper_color1)
    bin_image2 = cv2.inRange(hsv_frame, lower_color2, upper_color2)
    bin_image = cv2.bitwise_or(bin_image1, bin_image2)

    # オープニング・クロージングによるノイズ除去
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    oc = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, element8)
    oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8)

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(
        oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 一番大きい領域を指定する
        contours.sort(key=cv2.contourArea, reverse=True)
        cnt = contours[0]

        # 最小外接円を用いて円を検出する
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # 円が小さすぎたら円を検出していないとみなす
        if radius < minimum_radius:
            return None, None, None
        else:
            return oc, center, radius
    else:
        return None, None, None


def getCircle(hsv_frame, lower_color, upper_color):
    global minimum_radius
    # 指定した色範囲のみを抽出する
    bin_image = cv2.inRange(hsv_frame, lower_color, upper_color)

    # オープニング・クロージングによるノイズ除去
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    oc = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, element8)
    oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8)

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(
        oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 一番大きい領域を指定する
        contours.sort(key=cv2.contourArea, reverse=True)
        cnt = contours[0]

        # 最小外接円を用いて円を検出する
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # 円が小さすぎたら円を検出していないとみなす
        if radius < minimum_radius:
            return None, None, None
        else:
            return oc, center, radius
    else:
        return None, None, None


def main():  # メイン関数を定義する．
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # カメラの初期化

    global length_of_memory
    log_memory = np.zeros((length_of_memory, 10), dtype=float)
    log_index = 0
    log_exec = False

    state = 0

    if capture.isOpened() != True:
        print('camera device opening error')
        sys.exit()

    # ports = list_ports.comports(); #COMポートがひとつだけある場合はこれで見つかる．
    #Ser = serial.Serial(ports[0].device, 9600); #
    Ser = serial.Serial('COM4', 9600)  # COMポートが2個以上ある場合はこちらで指定
    if Ser.isOpen() != True:
        cv2.destroyAllWindows()  # 全てのOpenCVウィンドウを閉じる
        sys.exit()  # 終了

    desired_angle2 = 0

    start_time = time.time()
    keep_while_loop = True  # while loopを継続する
    while keep_while_loop:
        ret, frame = capture.read()  # カメラからデータを読み込むretには結果(True or False)が，frameには画像データがはいる
        windowsize = (800, 600)
        frame = cv2.resize(frame, windowsize)  # ウィンドウサイズを小さくする
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # HSVによる画像情報に変換
        blur = cv2.GaussianBlur(hsv, (9, 9), 0)  # ガウシアンぼかしを適用して、認識精度を上げる

        # 緑の円を見つける
        bin_green_image, center_g, radius_g = getCircle(
            blur, np.array([40, 40, 60]), np.array([90, 255, 255]))
        if bin_green_image is not None:
            # 見つかった円の上に緑の円を描画た円の上に緑の円を描画
            cv2.circle(frame, center_g, radius_g, (0, 255, 0), 2)
        # 赤の円を見つける
        bin_red_image, center_r, radius_r = getCircleDualThreshould(blur, np.array(
            [0, 120, 100]), np.array([30, 255, 255]), np.array([160, 120, 100]), np.array([180, 255, 255]))
        if bin_red_image is not None:
            # 見つかった円の上に赤の円を描画た円の上に緑の円を描画
            cv2.circle(frame, center_r, radius_r, (0, 0, 255), 2)
        # 青の円を見つける
        bin_blue_image, center_b, radius_b = getCircle(
            blur, np.array([90, 120, 100]), np.array([140, 255, 255]))
        if bin_blue_image is not None:
            # 見つかった円の上に青の円を描画た円の上に緑の円を描画
            cv2.circle(frame, center_b, radius_b, (255, 0, 0), 2)
        # 黄色の円を見つける
        bin_yellow_image, center_y, radius_y = getCircle(
            blur, np.array([20, 100, 80]), np.array([50, 255, 255]))
        if bin_yellow_image is not None:
            # 見つかった円の上に黄色の円を描画た円の上に緑の円を描画
            cv2.circle(frame, center_y, radius_y, (0, 255, 255), 2)

        if(state == 1):
            if (bin_red_image is not None) and (bin_green_image is not None):  # 赤と緑が見つかったら
                #print(center_g, center_r);
                x_g = float(center_g[0])  # 緑の円の中心のＸ座標をfloatにして代入
                y_g = float(center_g[1])  # 緑の円の中心のＹ座標をfloatにして代入
                x_r = float(center_r[0])  # 赤の円の中心のＸ座標をfloatにして代入
                y_r = float(center_r[1])  # 赤の円の中心のＹ座標をfloatにして代入
                dx = -(y_r - y_g)  # 角度計算のためのＸ変位
                dy = -(x_r - x_g)  # 角度計算のためのＹ変位
                angle1 = np.arctan2(dy, dx)*180.0/3.14159  # 角度計算
                angle1 = np.round(angle1, 1)  # 小数点2桁以下四捨五入
                print('angle1=', angle1, '[deg]')

                if angle1 > 47:
                    send_char = '3\r\n'  # 改行記号をつけて
                    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                    Ser.write(send_char_encoded)  # シリアル送信
                elif angle1 < 43:
                    send_char = '1\r\n'  # 改行記号をつけて
                    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                    Ser.write(send_char_encoded)  # シリアル送信
                else:
                    send_char = '2\r\n'  # 改行記号をつけて
                    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                    Ser.write(send_char_encoded)  # シリアル送信

                if (bin_blue_image is not None) and (bin_yellow_image is not None):  # 青と黄色が見つかったら
                    x_b = float(center_b[0])  # 青の円の中心のＸ座標をfloatにして代入
                    y_b = float(center_b[1])  # 青の円の中心のＹ座標をfloatにして代入
                    x_y = float(center_y[0])  # 黄色の円の中心のＸ座標をfloatにして代入
                    y_y = float(center_y[1])  # 黄色の円の中心のＹ座標をfloatにして代入
                    dx_by = -(y_y - y_b)  # 角度計算のためのＸ変位
                    dy_by = -(x_y - x_b)  # 角度計算のためのＹ変位
                    angle12 = np.arctan2(dy_by, dx_by)*180.0/3.14159  # 角度計算
                    angle12 = np.round(angle12, 1)  # 小数点2桁以下四捨五入
                    print('angle12=', angle12, '[deg]')
                    angle2 = angle12-angle1  # 仮の数値

                    if angle2 > 45 + 2.0:
                        send_char = '6\r\n'  # 改行記号をつけて
                        send_char_encoded = send_char.encode(
                            'utf-8')  # bytes型に変換
                        Ser.write(send_char_encoded)  # シリアル送信
                    elif angle2 < 45 - 2.0:
                        send_char = '4\r\n'  # 改行記号をつけて
                        send_char_encoded = send_char.encode(
                            'utf-8')  # bytes型に変換
                        Ser.write(send_char_encoded)  # シリアル送信
                    else:
                        send_char = '5\r\n'  # 改行記号をつけて
                        send_char_encoded = send_char.encode(
                            'utf-8')  # bytes型に変換
                        Ser.write(send_char_encoded)  # シリアル送信

                else:  # 青色と黄色の円のいずれかが見つからなかったら
                    send_char = '5\r\n'  # 改行記号をつけて
                    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                    Ser.write(send_char_encoded)  # シリアル送信

            else:  # 赤色と緑色の円のいずれかが見つからなかったら
                send_char = '2\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信
                send_char = '5\r\n'  # 改行記号をつけて
                send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
                Ser.write(send_char_encoded)  # シリアル送信

            if log_exec == True:
                current = time.time() - start_time
                log_input(log_memory, log_index, current, data2=current *
                          current, data3=angle1, data4=angle2)

                log_index = log_index + 1
                if log_index == length_of_memory:
                    log_exec = False
                    print('log finished!')

        kret = cv2.waitKey(1) & 0xFF  # 1ミリ秒の間キー入力を受け付ける
        if kret == ord('q'):  # qを押したら実行
            keep_while_loop = False  # while loopを脱出
        elif kret == ord('7'):  # 7を押したら実行
            desired_angle2 = 30.0  # [degree]
        elif kret == ord('8'):  # 8を押したら実行
            desired_angle2 = 0.0  # [degree]
        elif kret == ord('9'):  # 9を押したら実行
            desired_angle2 = -30.0  # [degree]
        elif kret == ord('r'):  # rを押したら実行
            log_index = 0
            log_exec = True  # logスタート
            print('log start!')
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
        elif kret == ord('s'):
            state = 1

        cv2.imshow('RGB', frame)  # frame画像を表示

    # while loopを抜けたあと
    send_char = '2\r\n'  # 改行記号をつけて
    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
    Ser.write(send_char_encoded)  # シリアル送信
    send_char = '5\r\n'  # 改行記号をつけて
    send_char_encoded = send_char.encode('utf-8')  # bytes型に変換
    Ser.write(send_char_encoded)  # シリアル送信
    Ser.close()  # シリアルを閉じる

    if log_index != 0:
        # ログデータをCSV形式で保存する．
        f = open('data.csv', 'w')
        csvWriter = csv.writer(f, lineterminator='\n')  # csvWriteの実装
        csvWriter.writerows(log_memory)  # CSVデータとして保存
        f.close()

    capture.release()  # カメラデバイスを解放
    cv2.destroyAllWindows()  # 全てのOpenCVウィンドウを閉じる


# Pythonのメイン関数
if __name__ == "__main__":
    main()
