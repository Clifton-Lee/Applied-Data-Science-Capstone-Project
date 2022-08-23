# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 22:32:34 2022

@author: Clift
"""
##############################################################################
#LIBRARIES
##############################################################################

#Web App
import requests
import streamlit as st
from streamlit_lottie import st_lottie

#Predictor Python file
import recommender as rec

#Other
#import time

##############################################################################



def main():
    
    st.set_page_config(page_title = "AdienceSound Recommendation System", page_icon = ":musical_note:", layout = "wide")
    
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    
    
    # ---- LOAD ASSETS ----
    lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_yosv8nkr.json")
    lottie_coding2 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_es4p9zph.json")
    
    # Header Section
    with st.container():
        st.subheader("Hi! Welcome to the :musical_note: AdienceSound Webapp :wave:")
        st.title("AdienceSound Recommendation System")
        st.write("This is a music recommendation system designed to assist travelers, tourists, and expats in determining what music would best suit them in the Asia Region.")
        
        
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("Take a photo :camera: & play your recommended songs :sound:")
            st.write("##")
            if st.button("Take Photo Here"):
                face_results = rec.detect_face()
                song_list = rec.recommend(face_results[0],face_results[1])
                st.success("Image Saved")
                st.success("Processing recommendations...")
                st.header("Recommended Songs")
                st.dataframe(song_list)
                st.subheader("[Listen to the songs here >](https://www.kkbox.com/intl/)")
                
        with right_column:
            st_lottie(lottie_coding, height = 400, key = "streaming")
            st.write("#")
            st_lottie(lottie_coding2, height = 400, key = "coding")
            
        
    
    
if __name__ == '__main__':
    main()
    