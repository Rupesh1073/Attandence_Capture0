import csv
import face_recognition_models

import cv2
import numpy as np

import os 
import glob
import pickle
from datetime import datetime
import time

video_capture_= cv2.VideoCapture(0)


my_img= face_recognition_models.load_image_file("C:\Coding299\Python\IMG_20230611_180147.jpg")
my_encoding= face_recognition_models.face_encodings(my_img)[0]

shivji_img= face_recognition_models.load_image_file("C:\Coding299\Python\Screenshot 2024-12-17 010025.png")
shivji_encoding= face_recognition_models.face_encodings(shivji_img)[0]

known_face_encoding =[
	my_encoding,
	shivji_encoding
]

known_faces_names=[
	"Roopesh",
	"Shiv ji"
]
students = known_faces_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(current_date+'.csv','w+',newline='')
lnwriter= csv.writer(f)

while True:
	_,frame= video_capture_.read()
	small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
	rgb_small_frame = small_frame[:,:,::-1]
	if s:
		face_locations=face_recognition_models.face_locations(rgb_small_frame)
		face_encoding=face_recognition_models.face_encodings(rgb_small_frame,face_locations)
		face_names=[]
		for face_encoding in face_encodings:
			matches = face_recognition_models.compare_faces(known_face_encoding,face_encoding,face_names)
			name=""
			face_distance = face_recognition_models.face_distance()
			best_match_index= np.argmin(face_distance,0)
			if matches[best_match_index]:
				name =known_faces_names[best_match_index]
			
			face_names.append(name)
			if name in known_faces_names:
				if name in face_names:
					students.remove(name)
					print(students)
					current_time =now.strftime("%H-%M-%S")
					lnwriter.writerow([name, current_time])
	cv2.imshow("attendence system",frame)
	if cv2.waitkey(1) & 0xFF == ord('q'):
		break

video_capture_.release()
cv2.destroyAllWindows()
f.close()