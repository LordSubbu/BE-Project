# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:08:19 2020

@author: Navin Subbu
"""

import pyrebase

config = {
    "apiKey" : "AIzaSyAx3rxrUmCOESoCdj3NP1x_ITyeC_YYjno",
    "authDomain" : "python-test-1235f.firebaseapp.com",
    "databaseURL" : "https://python-test-1235f.firebaseio.com",
    "projectId" : "python-test-1235f",
    "storageBucket" : "python-test-1235f.appspot.com",
    "messagingSenderId" : "707011126206",
    "appId" : "1:707011126206:web:93df56c96e28b114b48fef",
    "measurementId" : "G-8YHDDWVLP7"
    }



def firebase_store(path,image_name) :
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    
    path_on_cloud = "Car Recorded{0}/{1}".format(path,image_name)
    path_local = "detected/{0}.".format(image_name) 
    storage.child(path_on_cloud).put(path_local)