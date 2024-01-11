import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance

# 分類器の指定
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:/Users/keisu/Downloads/haarcascade_frontalface_alt2.xml")
face_parts_detector = dlib.shape_predictor('gaze_cv/shape_predictor_68_face_landmarks.dat')

# EAR（目の開閉判別式）の定義
def calc_ear(eye):
    print(eye)
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
    print(face_parts)
    # EARの計算
    left_eye_ear = calc_ear(face_parts[42:48])
    #cv2.putText(rgb, "left eye EAR:{} ".format(round(left_eye_ear, 3)), 
    #    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    right_eye_ear = calc_ear(face_parts[36:42])
    #cv2.putText(rgb, "right eye EAR:{} ".format(round(right_eye_ear, 3)), 
    #    (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    # 口の大きさの計算
    mouse_size = calc_mou(face_parts[60:68])
    #cv2.putText(rgb, "mouse size:{} ".format(round(mouse_size, 3)), 
    #    (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 目の開閉の判別
    if left_eye_ear > 0:
        eye_l = 1
    else:
        eye_l = 0
        
    if right_eye_ear > 0:
        eye_r = 1
    else:
        eye_r = 0
    
    # 口の開閉の判別
    if mouse_size > 0.07:
        mouse_open = 1
    else:
        mouse_open = 0
    judged =[eye_l, eye_r, mouse_open]
    return judged


# ウィンドウの準備
cv2.namedWindow("face_and_eye")

# 変換処理ループ
while True:
    tick = cv2.getTickCount()

    ret, rgb = cap.read()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    if len(faces) == 1:
        
        x, y, w, h = faces[0, :]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = dlib.rectangle(x, y, x + w, y + h)
        face_parts = face_parts_detector(gray, face)
        face_parts = face_utils.shape_to_np(face_parts)

        open =  judge(face_parts)
        
    #    cv2.putText(rgb, "l eye op:{}".format(round(open[0], 1)), (200, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 1, cv2.LINE_AA)
    #    cv2.putText(rgb, "r eye op:{}".format(round(open[1], 1)), (200, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 1, cv2.LINE_AA)
    #    cv2.putText(rgb, "mouse_open:{}".format(round(open[2], 3)), (200, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 1, cv2.LINE_AA)
        
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    #cv2.putText(rgb, "FPS:{} ".format(int(fps)), 
    #    (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # フレーム作成
    cv2.imshow("face_and_eye", rgb)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

# 終了処理
cap.release()
cv2.destroyAllWindows()