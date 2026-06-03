import cv2
import time
import os
import threading
import numpy as np
from PIL import Image
import serial
import pygame
import math
import random


faceCascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX
ser=serial.Serial('/dev/ttyUSB0',9600)

pygame.mixer.init()
pygame.mixer.music.load("/home/mech-user/Desktop/東大３S（履修・課題等）/advanstprogram/ブザー音.mp3")

red_frame = cv2.imread("Red frame.jpg")
red_frame_resized = cv2.resize(red_frame, (640, 480))
wanted=cv2.imread("WANTED.jpeg")

# 名前とIDの対応関係を管理する辞書
name_to_id = {}
save_again=0


imagefile1 = [os.path.join("顔認証（グレー）/",f) for f in os.listdir("顔認証（グレー）/")]
if not imagefile1:
    current_id = 0  # 登録データがないときIDを初期化
else:
    maxid=0
    for images1 in imagefile1:
        file_name1 = os.path.basename(images1)
        id_str1 = file_name1.split(".")[2]  # ファイル名からID部分を取得
        id1 = int(id_str1)
        if id1 > maxid:
            maxid=id1
    current_id = maxid #登録データがあるとき初期IDはその最大値（追加する際プラス１）
    
id_to_name=["Unknown"]

registered=False
mouse_break=False
progress=0
rect_color = (0, 255, 0)
unknown_color=None

# カメラを起動する関数
def start_camera():
    # V4L2を使用してカメラを起動してキャプチャを開始
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# カメラを停止する関数
def stop_camera(cap):
    # キャプチャを停止してウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

# 距離が指定範囲内かどうかを判断する関数（distanceは距離センサーで取得）
def is_distance_in_range(distance, min_distance, max_distance):
    return min_distance <= distance <= max_distance

# 顔の検出と画像のキャプチャを行う関数
def capture_face(cap, directory, directory1, name, min_distance, max_distance):
    count = 0
    global registered,progress
    registered=False #二人目を登録するとき初期化
    progress=0 #二人目を登録するとき初期化
    distance=0
    while True:
        ret, frame = cap.read()
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,     
                                             scaleFactor=1.2,
                                             minNeighbors=5,     
                                             minSize=(20, 20))

        # 距離を取得（距離センサーで）
        serialCommand="d"
        ser.write(serialCommand.encode())
        line = ser.readline()
        line_str = line.decode('utf-8').strip()
        
        try:
            distance = float(line_str)
            print(f"distance: {distance}")
        except ValueError as e:
            continue

        # 距離が指定範囲内にあり、かつ顔が検出された場合に画像をキャプチャ
        if is_distance_in_range(distance, min_distance, max_distance) and len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                capture_thread = threading.Thread(target=save_image, args=(roi_gray, roi_color, directory, directory1,  name, count))
                capture_thread.start()

                count += 1  # 撮影した画像の数を増やす
                if progress <50:
                    progress += 1

                if count>=50 and not registered:
                    print("Successfully registered!")
                    registered=True
                

        if count >= 50:
            break
        time.sleep(0.2)

# 画像を保存する関数
def save_image(roi_gray, roi_color, directory, directory1, name, count):
    global current_id, save_again

    if save_again!=1:
        image_files = os.listdir(directory)
        id_name_pairs = [(int(file.split('.')[2]), str(file.split('.')[1])) for file in image_files if file.endswith('.jpg')]
        sorted_id_name_pairs=sorted(id_name_pairs, key=lambda x:x[0])
        for i in range (0, len(sorted_id_name_pairs)):
            if sorted_id_name_pairs[i][1] not in id_to_name:
                name_to_id[sorted_id_name_pairs[i][1]]=sorted_id_name_pairs[i][0]
                id_to_name.append(sorted_id_name_pairs[i][1])
        save_again=1
            

    
    if name not in name_to_id:
        current_id += 1
        name_to_id[name] = current_id
        id_to_name.append(str(name))

    id = name_to_id[name]
    
    file_name_gray = f"{directory}gray_face.{name}.{id}.{count + 1}.jpg"
    cv2.imwrite(file_name_gray, roi_gray)
    file_name_color = f"{directory1}color_face.{name}.{id}.{count + 1}.jpg"
    cv2.imwrite(file_name_color, roi_color)




# Webカメラの動画を表示する関数
def display_video(cap):
    global registered, progress, mouse_break, rect_color
    mouse_break=False#二人目を登録するとき初期化
    rect_color=(0,255,0)
    def on_mouse(event, x, y, flags, param):
        global mouse_break, rect_color
        if event == cv2.EVENT_LBUTTONDOWN:
            # クリックされた座標が特定の領域内かどうか判定
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH)-140 <= x <= cap.get(cv2.CAP_PROP_FRAME_WIDTH)-40 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100 <= y <= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-20:
                mouse_break = True
        if event == cv2.EVENT_MOUSEMOVE:
            # 移動した座標が特定の領域内かどうか判定
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH)-140 <= x <= cap.get(cv2.CAP_PROP_FRAME_WIDTH)-40 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100 <= y <= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-20:
                rect_color=(0,140,255)
            else:
                rect_color=(0,255,0)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:  # もしフレームが空の場合
            continue  # 次のループに進む
        frame1 = cv2.flip(frame, 1)  # 画像を左右反転
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,     
                                             scaleFactor=1.2,
                                             minNeighbors=5,     
                                             minSize=(20, 20))
        
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if registered:
            cv2.putText(frame1, 'Registration Completed', (100, 50), font, 1, (0, 140, 255), 3)
            cv2.setMouseCallback("Web Camera", on_mouse)
            cv2.putText(frame1, 'Close', (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))-130, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-50), font, 1, rect_color, 2)
            cv2.rectangle(frame1, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))-140, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-100), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))-40, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-20), rect_color, 2)


        # プログレスバーの描画
        progress_angle = int(360 * progress / 50)  # progressに応じた角度を計算
        cv2.ellipse(frame1, (320, 260), (100, 125), 0, -90+progress_angle, 270,(0, 255, 255), 2)
        cv2.ellipse(frame1, (320, 260), (100, 125), 0, -90, -90+progress_angle, (30, 255, 0), 2)


        cv2.imshow("Web Camera", frame1)
        c = cv2.waitKey(2)
        if c == 27 or mouse_break: # Escを押すか範囲内を左クリック
            break


#キー入力を選択
def wait_key():
    while True:
        key = input("キーを押してください: ")
        if key == 'r' or key == 't':
            return key
        else:
            print("無効なキーが押されました。'r' または 't' を押してください。")

#入力した写真でトレーニング

def training_image(trained_file):
    imagefile = [os.path.join(trained_file,f) for f in os.listdir(trained_file)]     
    samples = []
    ids = []

    for images in imagefile:
        file_name = os.path.basename(images)
        id_str = file_name.split(".")[2]  # ファイル名からID部分を取得

        try:
            id = int(id_str)
        except ValueError:
            print(f"ID無効: {images}")
            continue

        PIL_img = Image.open(images).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')

        faces = faceCascade.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    if len(samples) != len(ids):
        print("サンプル数とIDの数が不一致")
        return [], []

    return samples, ids



#顔照合
def face_detect(cap,directory1):
    unknown_count=0
    transparency=0
    global mouse_break, rect_color, unknown_color
    mouse_break=False #初期化
    rect_color=(0,255,0)
    unknown_color=None
    frame1=None
    wanted_r=None
    unknownfaces=[]
    def on_mouse1(event, x, y, flags, param):
        global mouse_break, rect_color, unknown_color
        nonlocal wanted_r
        if event == cv2.EVENT_LBUTTONDOWN:
            # クリックされた座標が特定の領域内かどうか判定
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH)-140 <= x <= cap.get(cv2.CAP_PROP_FRAME_WIDTH)-40 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100 <= y <= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-20:
                mouse_break = True

            else:
                if len(unknownfaces)>0:
                    for i in range(0,len(unknownfaces)): #Unknownをクリックすると指名手配ポスター
                        if unknownfaces[i][0]<=x<=unknownfaces[i][0]+unknownfaces[i][2] and unknownfaces[i][1]<=y<=unknownfaces[i][1]+unknownfaces[i][3]:
                            unknown_color=frame1[unknownfaces[i][1]:unknownfaces[i][1]+unknownfaces[i][3],unknownfaces[i][0]:unknownfaces[i][0]+unknownfaces[i][2]]
                            wanted_r=wanted.copy()
                            cv2.putText(wanted_r, "$"+str(100*(random.randint(5,1000))), (56,255), font, 1.2, (10,0,10), 3)
        if event == cv2.EVENT_MOUSEMOVE:
            # 移動した座標が特定の領域内かどうか判定
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH)-140 <= x <= cap.get(cv2.CAP_PROP_FRAME_WIDTH)-40 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100 <= y <= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-20:
                rect_color=(0,140,255)
            else:
                rect_color=(0,255,0)

    cv2.putText(wanted, "Reward!", (63,297), font, 1, (10,0,10), 3)
    
    while True:
        unknownfaces=[]
        ret, frame = cap.read()
        if not ret or frame is None:  
            continue  
        frame1 = cv2.flip(frame, 1) 
        gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(0.1*cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(0.1*cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

        if(len(faces)<=0):
                if(unknown_count > 0):
                    unknown_count-=2
                unknownfaces=[]
      
        for (x,y,w,h) in faces: 
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                    

            try:
                 
                if 35< 100-confidence :
                    collated_name = id_to_name[id]
                    if(unknown_count > 0):
                        unknown_count-=3
                    
                    confidence = "  {0}%".format(round(100 - confidence))
                    cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame1, "Certification:"+str(collated_name), (x-15,y-5), font, 1, (0,255,0), 2)
                    cv2.putText(frame1, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
                    # カラー写真を読み込んで透過処理で重ねる
                    file_name_color = f"{directory1}color_face.{collated_name}.{id}.1.jpg" 
                    if os.path.exists(file_name_color):
                        color_image = cv2.imread(file_name_color)
                        if color_image is not None:
                            color_image_resized = cv2.resize(color_image, (150, 150))
                            
                            # 顔の長方形の範囲外の右下に配置する
                            overlay_x = x + w  
                            overlay_y = y + h  

                            if overlay_x + color_image_resized.shape[1] <= frame1.shape[1]:  
                                start_x = overlay_x
                            else:
                                start_x = frame1.shape[1] - color_image_resized.shape[1]
                            if overlay_y + color_image_resized.shape[0] <= frame1.shape[0]:  
                                start_y = overlay_y
                            else:
                                start_y = frame1.shape[0] - color_image_resized.shape[0]

                            end_x = start_x + color_image_resized.shape[1]
                            end_y = start_y + color_image_resized.shape[0]
                            blended_image = cv2.addWeighted(frame1[start_y:end_y, start_x:end_x], 0.5, color_image_resized, 0.5, 0)
                            frame1[start_y:end_y, start_x:end_x] = blended_image
                else:
                    unknownfaces.append([x,y,w,h])
                    collated_name = "unknown"
                    if(unknown_count<180):
                        unknown_count+=1
                    
                    confidence = "  {0}%".format(round(100 - confidence))
                    cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,0,255), 2)
                    cv2.putText(frame1, str(collated_name), (x+5,y-5), font, 1, (0,0,255), 2)
                    cv2.putText(frame1, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            except Exception as e:
                print(f"一時的検出エラー: {str(e)}")
                continue


        cv2.setMouseCallback("security camera", on_mouse1)
        if unknown_color is not None and wanted_r is not None:
            wanted_r[85:212, 63:190]=cv2.resize(unknown_color, (127,127))
            cv2.imshow("Unknown face",wanted_r)

        if(unknown_count > 60):
            serialCommand="b"
            ser.write(serialCommand.encode())

            cv2.putText(frame1, 'Unknown face detected!!', (100, 50), font, 1, (0, 0, 255), 3)

            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
            transparency+=1
            tr=0.10*(1-math.cos(transparency*math.pi/15))
            blended_frame = cv2.addWeighted(frame1, 1-tr, red_frame_resized, tr, 0)
            frame1 = blended_frame
        else:
            serialCommand="r"
            ser.write(serialCommand.encode())

            pygame.mixer.music.stop()
            transparency=0

        
        cv2.putText(frame1, 'Close', (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))-130, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-50), font, 1, rect_color, 2)
        cv2.rectangle(frame1, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))-140, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-100), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))-40, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-20), rect_color, 2)
            
        

        cv2.imshow("security camera",frame1)


        k = cv2.waitKey(2) & 0xff 
        if k == 27 or mouse_break: #Escキーを押すか左クリック
            break


    


    

# メイン関数
def main():
    while True:
        #名前を予め入力
        name = input('Input your name (alphabet only):')
        print(f'Your name: {name}')
        directory = "顔認証（グレー）/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory1 = "顔認証（カラー）/"
        if not os.path.exists(directory1):
            os.makedirs(directory1)
        trained_file="顔認証（グレー）/"
        # 距離の閾値を設定
        min_distance = 40  # 最小距離（センチメートル）
        max_distance = 50  # 最大距離（センチメートル）



        # カメラを起動
        cap = start_camera()
        
        # 顔の検出と画像のキャプチャを行うスレッドを開始
        capture_process = threading.Thread(target=capture_face, args=(cap, directory, directory1, name, min_distance, max_distance))
        capture_process.start()

        # Webカメラの動画を表示するスレッドを開始
        display_process = threading.Thread(target=display_video, args=(cap,))
        display_process.start()

        # スレッドが終了するまで待機
        capture_process.join()
        display_process.join()

        stop_camera(cap)

        print("別の顔を登録=>Press r")
        print("顔の学習・照合に進む=>Press t")
        key = wait_key()

        if key == 'r':
            continue
        elif key == 't':
            break

    directory2 = "顔学習/"
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    print ("\n 顔を学習しています。。。")
    print(id_to_name)
    print(name_to_id)
    faces,ids = training_image(trained_file)
    recognizer.train(faces, np.array(ids))

    recognizer.write('顔学習/顔学習.yml')

    print(f"\n {len(np.unique(ids))}人の顔登録完了")

    recognizer.read('顔学習/顔学習.yml')

    cap=start_camera()

    face_detect(cap,directory1)

    stop_camera(cap)

    yml_file = '顔学習/顔学習.yml'
    if os.path.exists(yml_file):
        os.remove(yml_file)
        print(f"\n {yml_file} を削除しました。")

    print("\n 登録した写真をすべて削除しますか？\n削除=>Press y　保存したまま終了=>Press n")
    while True:
        key1= input("キーを押してください: ")
        if key1 == 'y':
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"{file_path} を削除しました。")
                except Exception as e:
                    print(f"ファイルの削除エラー: {e}")
            for file_name in os.listdir(directory1):
                file_path = os.path.join(directory1, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"{file_path} を削除しました。")
                except Exception as e:
                    print(f"ファイルの削除エラー: {e}")
            break
        elif key1 == 'n':
            break

        else:
            print("無効なキーが押されました。'y' または 'n' を押してください。")

    

if __name__ == "__main__":
    main()
