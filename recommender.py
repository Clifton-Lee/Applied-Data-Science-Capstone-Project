# -*- coding: utf-8 -*-
"""
Applied Data Science Final Project

@author: Clifton Lee, ID: 620040802
"""
###############################################################################
#LIBRARIES 
###############################################################################

import shutup
shutup.please()

# pandas and numpy
import pandas as pd
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

# model loader
from joblib import load


#Facial recognition
import cv2
import tensorflow as tf
import numpy as np
from deepface import DeepFace
import face_recognition

#other
import time

###############################################################################
#FACIAL RECOGNITION
###############################################################################

#Load the gender model
gender_model = tf.keras.models.load_model('gender_model25.h5')

#Load age model
age_model = tf.keras.models.load_model('age_model50.h5')

def detect_face():
        
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    
    while True:
        try:
            # Grab a single frame of video
            check, frame = webcam.read()
            
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]
            
            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Get coordinates of face location
            for top, right, bottom, left in face_locations:
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Display the resulting image
            cv2.imshow('Capturing', frame)
            
            key = cv2.waitKey(1)
            
            if key == ord('s'): 
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                print("Image saved!")
                print("Processing image...")
                
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
        
    new_frame=cv2.imread('saved_img.jpg')
    #gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    image= new_frame
    try:
        face_locations = face_recognition.face_locations(image)
        for top, right, bottom, left in face_locations:
            image2 = new_frame[top: bottom, left: right]
            cv2.rectangle(new_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
    except:
        image2 = image
        
    time.sleep(3)
    # predict ages and genders of the detected faces
    img2= cv2.resize(image2, (227, 227))
    img2=np.array([img2]).reshape((1, 227,227,3))
    #gender_results = gender_model.predict(img2)
    #age_results = age_model.predict(img2)
    #predicted_gender = gender_results[0][0]
    #gen="Male" if predicted_gender > 0.5 else "Female"
    ages = [['0-5'],['6-12'],['13-20'],['21-29'],['30-39'],['40-50'],['51-62'],['63-100']]
    obj = DeepFace.analyze(img_path="saved_img.jpg",enforce_detection= False)
    for age_list in ages:
        numbers = age_list[0].split('-')
        if (int(obj['age']) >= int(numbers[0])) and (int(obj['age']) <= int(numbers[1])):
            predicted_ages = age_list[0]
    
    gen2 = {'Man': 'M', 'Woman': 'F', 'Male': 'M','Female': 'F'}
    # draw results
    pred=""
    pred=predicted_ages + " , " + gen2[obj['gender']]
    #print(pred)
    cv2.putText(new_frame, pred,(face_locations[0][3],face_locations[0][0]) , cv2.FONT_HERSHEY_SIMPLEX,0.7, (2, 255, 255), 2)


    cv2.imshow('Gender and age', new_frame)
    cv2.waitKey(5000)  # ESC key press
    cv2.destroyAllWindows()

    return predicted_ages,gen2[obj['gender']]
    
#face_results = detect_face()

###############################################################################
#Recommender System 
###############################################################################

      
#read in the datasets
songs_encode = pd.read_csv('songs_encode.csv',index_col=0)
songs = pd.read_csv('songsupdated.csv', index_col=0)

#Create recommendor function 
def recommend(age_range,gender):
    [['0-5'],['6-12'],['13-20'],['21-29'],['30-39'],['40-50'],['51-62'],['63-100']]
    age_map = {'0-5':(0,5), '6-12':(6,12), '13-20':(13,20), '21-29':(21,29), '30-39':(30,39), '40-50':(40,50), '51-62':(51,62),'63-100':(63,100)}
    gender_map = {'m': 'male', 'f': 'female'}
    
    rec_songs = songs_encode[(songs['bd']>= age_map[age_range][0]) & (songs['bd'] <= age_map[age_range][1]) & (songs['gender'] == gender_map[gender.lower()])]
     
    # load the model from disk
    loaded_model = load(open('finalized_model.sav', 'rb'))
    rec_songs['predictions'] = loaded_model.predict(rec_songs)
    song_list = rec_songs[rec_songs['predictions'] == 1]
    the_list = songs.loc[list(song_list.index),['artist_name','name']]
    final_list =  the_list.value_counts()[:10].to_frame()
    final_list.reset_index(inplace = True)
    final_list = final_list.rename(columns = {0: 'mostly_played','name': 'song_name'})
    final_list.index += 1
    return final_list
    #return final_list

#call function 
#recommend(face_results[0],face_results[1])
