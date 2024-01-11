# coding:utf-8

import argparse
import cv2
from tracking_system import FaceLandmarkManager
from tracking_system import EyeSystemManager


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
        ret, img = cap.read()

        # 顔のランドマークリストを取得
        face_manager.clear_face_landmark_list()
        face_manager.detect_face_landmark(img)
        face_landmark_list = face_manager.get_face_landmark_list()

        # 目領域の取得
        for face_landmark in face_landmark_list:
            eye_manager = EyeSystemManager()
            eye_manager.detect_eye_region(face_landmark)

            # 虹彩領域の取得
            right_iris, left_iris = eye_manager.detect_iris_info(img)

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

            print("position_leftX:", position_leftX)
            print("position_rightX:", position_rightX)
            print("position_leftY:", position_leftY)
            print("position_rightY:", position_rightY)
            
            
            

        # 結果の表示
        cv2.imshow('readme_img', img)

        # 'q'が入力されるまでカメラ画像を表示し続ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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


if __name__ == '__main__':
    args_value = get_args()
    main(args_value)