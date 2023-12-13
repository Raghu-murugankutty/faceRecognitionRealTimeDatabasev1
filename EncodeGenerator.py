import cv2
import face_recognition
import pickle
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://faceattendancerealtime-790e0-default-rtdb.firebaseio.com/',
    'storageBucket': 'faceattendancerealtime-790e0.appspot.com'
})

# Importing student images from the local folderPath for face encodings
folderPath = 'Images'
pathList = os.listdir(folderPath)
print('pathList',pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    # Splitting student id from the image name eg: 123123.png, extracting only student id = 123123
    studentIds.append(os.path.splitext(path)[0])
    print('studentIds', studentIds)

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

# Encoding the student images using cv2 library
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
    # before encoding we have to convert the image color to BGR to DGB
    # Opencv uses BGR, Face recognition library uses RGG
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # -----> now we are encoding the RGB images of the student in a loop
        # -----> storing that encoded values in the "encodeList"
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
print('encodings started...')
# -----> here we are invoking the "findEncodings" function and passing the argument as student image list
# -----> now we have the encoded values in the variable ------->  #encodeListKnown
print('----------encodings started----------')
encodeListKnown = findEncodings(imgList)
# -----> we want to know which encoding values corresponds to which student ids
encodeListKnownWithIds = [encodeListKnown, studentIds]
print('----------encodings completed----------')
print('student images encoded matrix list', encodeListKnownWithIds)
# -----> Saving the encodings in a file, so later we can pull this file for comparison with realtime face images
file = open('EncodeFile.p', 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print('file saved')
