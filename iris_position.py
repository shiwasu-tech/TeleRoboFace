# coding:utf-8

import argparse
import cv2
from tracking_system import FaceLandmarkManager
from tracking_system import EyeSystemManager
from scipy.spatial import distance
import numpy as np
import serial
import time

#シリアル通信の設定
#Ser = serial.Serial('COM12',9600,timeout=3)

def Serialsend(serial_send_data):
    Ser.write(serial_send_data.encode())

def get_args():
    """
    コマンドライン引数の処理
    :return args_value: 受け取ったコマンドライン引数の値
    """
    # 使用するWebカメラの番号を設定するコマンドライン引数の作成
    parser = argparse.ArgumentParser(description='Iris tracking system')
    help_msg = 'Set web-cam number.'
    parser.add_argument('CAM_NUM', default=0, nargs='?', type=int, help=help_msg)

    # コマンドライン引数の受け取り
    args = parser.parse_args()

    return args


def get_iris_from_cam(cam_no):
    """
    Webカメラから顔画像を取得し、顔から虹彩を検出する
    :param cam_no: 使用するWebカメラ番号
    """

    cap = cv2.VideoCapture(cam_no)
    face_manager = FaceLandmarkManager()

    # カメラ画像の表示('q'で終了)
    while True:
        for i in range(10):
            ret, img = cap.read()
            

        # 顔のランドマークリストを取得
        face_manager.clear_face_landmark_list()
        face_manager.detect_face_landmark(img)
        face_landmark_list = face_manager.get_face_landmark_list()
        
        status_string = ['0','0','0','0','0','0','0']
        
        if face_landmark_list:
            eye_potision_status = get_eye_status(face_landmark_list,img)
            eye_mouse_open_status = judge(face_landmark_list)
            status = np.append(eye_mouse_open_status,eye_potision_status)
            
            if(status[0] == 1):
                status_string[0] = "左目-開"
            else:
                status_string[0] = "左目-閉"
        
            if(status[1] == 1):
                status_string[1] = "右目-開"
            else:
                status_string[1] = "左目-閉"
        
            if(status[2] == 1):
                status_string[2] = "口-開"
            else:
                status_string[2] = "口-閉"
        
            status_string[3] = "左目x-"+str(status[3])
            status_string[4] = "左目y-"+str(status[4])
            status_string[5] = "右目x-"+str(status[5])
            status_string[6] = "右目x-"+str(status[6])
        
        
        
        
        print(status_string)
        
        
        #シリアル通信の送信部
        #Serialsend(str(status))

        # 結果の表示
        cv2.imshow('readme_img', img)

        # 'q'が入力されるまでカメラ画像を表示し続ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)

    # 後処理
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    """
    メイン関数
    :param args: コマンドライン引数の値
    :return:
    """
    # 引数で指定された番号のWebカメラを用いて虹彩跡実施
    cam_num = args.CAM_NUM
    get_iris_from_cam(cam_num)

def get_eye_status(landmark_list,image):
  # 目領域の取得
    for face_landmark in landmark_list:
        eye_manager = EyeSystemManager()
        eye_manager.detect_eye_region(face_landmark)

        # 虹彩領域の取得
        right_iris, left_iris = eye_manager.detect_iris_info(image)

        # 虹彩の位置から目の中での位置を-1から1の間に変換
          #左右の判定
          #左目 
        center_leftX_coordinates = left_iris['center'][0]
        top_leftX_coordinates = face_landmark[42][0]
        bottom_leftX_coordinates = face_landmark[45][0]
        middle_leftX = (top_leftX_coordinates + bottom_leftX_coordinates)/2
        lange_leftX = (bottom_leftX_coordinates - top_leftX_coordinates)/2
        position_leftX = (center_leftX_coordinates -middle_leftX)/lange_leftX
          #右目
        center_rightX_coordinates = right_iris['center'][0]
        top_rightX_coordinates = face_landmark[36][0]
        bottom_rightX_coordinates = face_landmark[39][0]
        middle_rightX = (top_rightX_coordinates + bottom_rightX_coordinates)/2
        lange_rightX = (bottom_rightX_coordinates - top_rightX_coordinates)/2
        position_rightX = (center_rightX_coordinates -middle_rightX)/lange_rightX
          #上下の判定
          #左目
        center_leftY_coordinates = left_iris['center'][1]
        top_leftY_coordinates = face_landmark[43][1]
        bottom_leftY_coordinates = face_landmark[47][1]
        middle_leftY = (top_leftY_coordinates + bottom_leftY_coordinates)/2
        lange_leftY = (bottom_leftY_coordinates - top_leftY_coordinates)/2
        position_leftY = (center_leftY_coordinates -middle_leftY)/lange_leftY
          #右目
        center_rightY_coordinates = left_iris['center'][1]
        top_rightY_coordinates = face_landmark[37][1]
        bottom_rightY_coordinates = face_landmark[41][1]
        middle_rightY = (top_rightY_coordinates + bottom_rightY_coordinates)/2
        lange_rightY = (bottom_rightY_coordinates - top_rightY_coordinates)/2
        position_rightY = (center_rightY_coordinates -middle_rightY)/lange_rightY

        '''
        print("position_leftX:", position_leftX)
        print("position_rightX:", position_rightX)
        print("position_leftY:", position_leftY)
        print("position_rightY:", position_rightY)
        '''
        
        eye_potision = [[position_leftX,position_leftY],[position_rightX,position_rightY]]
        
        return eye_potision

# EAR（目の開閉判別式）の定義
def calc_ear(eye):
    #print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = ((A + B) / (2.0 * C) - 0.20) / 0.15
    return round(eye_ear, 3)

# 口の開き具合の判別式の定義
def calc_mou(mou):
    D = distance.euclidean(mou[1], mou[7])
    E = distance.euclidean(mou[2], mou[6])
    F = distance.euclidean(mou[3], mou[5])
    G = distance.euclidean(mou[0], mou[4])
    mouse = (D + E + F) / (3.0 * G)
    return round(mouse, 3)

# 開いているかの判断
def judge(face_parts):
    #print(face_parts)

    # EARの計算
    left_eye_ear = calc_ear(face_parts[0][42:48])
    #cv2.putText(rgb, "left eye EAR:{} ".format(round(left_eye_ear, 3)), 
    #    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    right_eye_ear = calc_ear(face_parts[0][36:42])
    #cv2.putText(rgb, "right eye EAR:{} ".format(round(right_eye_ear, 3)), 
    #    (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    # 口の大きさの計算
    mouse_size = calc_mou(face_parts[0][60:68])
    #cv2.putText(rgb, "mouse size:{} ".format(round(mouse_size, 3)), 
    #    (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 目の開閉の判別
    if left_eye_ear > 0.05:
        eye_l = 1
    else:
        eye_l = 0
        
    if right_eye_ear > 0.05:
        eye_r = 1
    else:
        eye_r = 0
    
    # 口の開閉の判別
    if mouse_size > 0.1:
        mouse_open = 1
    else:
        mouse_open = 0
    judged =[eye_l, eye_r, mouse_open]
    return judged

#Serial通信
def serial_send(facial_data):
  print("a")



if __name__ == '__main__':
    args_value = get_args()
    main(args_value)
    
Ser.close()