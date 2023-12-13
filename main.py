import pickle
import numpy as np
import cv2
import os
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://faceattendancerealtime-790e0-default-rtdb.firebaseio.com/',
    'storageBucket': 'faceattendancerealtime-790e0.appspot.com'
})
bucket = storage.bucket()
# Open the default camera (usually index 0)
import face_recognition

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    
# Set the frame dimensions
cap.set(3, 640)
cap.set(4, 480)
imgBackground = cv2.imread('Resources/background.png')

# importing static mode template  images from local folders
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
print('modePathList', modePathList)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

print('imgModeList',imgModeList)

#
# 5 ----> we encoded the images in EncodeGenerator.py ------> then loading the encoded file
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds  = encodeListKnownWithIds
print('-----studentIds loaded from file--------',studentIds)
print('-----encoded values loaded from file--------',encodeListKnown)

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, img = cap.read()
    # resizing the captured images, so that we can place in static frame
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # generating the encoding values for the realtime webcam images
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)


    if not success:
        print("Error: Failed to read frame.")
        break

    # cv2.imshow("Webcam", img)

    #
    imgBackground[162:162 + 480, 55:55 + 640] = img # Here we are placing the realtime captured image on top of imgBackground
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType] # Here we are placing the static mode template images on top of imgBackground
    if faceCurFrame:
        # inside loop we are comparing the encoded values of both webcam images and student faces already encoded
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print('--------matches--------:', matches) # -------> matches [False, True, False, False, False, False]
            print('--------faceDis--------:', faceDis) # lesser face distance more match
            #-------> faceDis [0.72983823 0.45094869 0.60408934 0.80314018 0.75740034 0.84982606]

            # -------> we know that minimum faceDis is the match ---> taking the min value from the list
            matchIndex = np.argmin(faceDis)
            print('---------Match Index--------:', matchIndex)

            if matches[matchIndex]:
                # print('Known face detected')
                # print(studentIds[matchIndex])

                # code to create the bounding box on the face pic
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 # previously we reduced the size, now we are multiplying with 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]

                if  counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter !=0:
            if counter == 1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)
                # get the image from storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                   "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)

                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                # # update data of attendance
                # ref = db.reference(f'Students/{id}')
                # studentInfo['total_attendance'] += 1
                # ref.child('total_attendance').set(studentInfo['total_attendance'])

        if modeType != 3:
            if 10<counter<20:
                modeType = 2
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if counter <=10:
                cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(id), (1006, 493),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                offset = (414 - w) // 2
                cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

            counter+=1

            if counter >= 20:
                counter = 0
                modeType = 0
                studentInfo = []
                imgStudent = []
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)
    key = cv2.waitKey(1)  # Wait for a key press for 1 millisecond

    if key == ord('q'):  # Exit the loop if 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
