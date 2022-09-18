# Get predictions for each frame either from camera or video

# This is the main file that run GUI and get predictions

# import the necessary libraries
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from numpy import expand_dims
import numpy as np
from pickle5 import load
from kivy.lang import Builder
from kivy.graphics import Color, Line
from kivy.uix.floatlayout import FloatLayout
from playsound import playsound
from kivy.uix.camera import Camera

from pandas import DataFrame

import cv2


from PIL import Image   as Imagepil



# load the pickled model
with open('./speed_detector.pkl', 'rb') as file:
    model = load(file)










# create CamApp class

class CamApp(App):

  
  

    def build(self):
    

     

        

       # this is the image to show the sign board
        self.speedimage = Image(size_hint= (1.0, 1.0), size=(224,224))      
    
       
       

       
         
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        
        
       
        
        
       
        #self.capture = cv2.VideoCapture(0)  # uncomment this line to reach camera
        
        # Frames are obtained from  the video
        self.capture = cv2.VideoCapture("video2.mp4")
        
        # show speed image on the layout
        layout.add_widget(self.speedimage)
        
        # update frames 
        Clock.schedule_interval(self.update, 1.0/10)
        return layout





    def update(self, dt):
    
    
       
        # get frames
        ret, frame = self.capture.read()

        buf1 = cv2.flip(frame, 0)
      
        buf = buf1.tobytes()
        
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        
        # display image from the texture
        self.img1.texture = texture1

        # resize image for the model input
        resized_img = cv2.resize(frame, (224,224), interpolation = cv2.INTER_AREA)


      

        
        
        
        def image_preprocessing(resized_image):
         """
          This function preprocesses the input image for MobileNetV2
         """
         pic_array = np.asarray( resized_image, dtype="int32" )
         image_batch = expand_dims(pic_array, axis=0)
         processed_image = (image_batch/127.5)-1
         return processed_image
        
        
        classes = ['mixed', 'speed30zone']
        
        def image_classification(preprocessed_image, model):
          probs = model.predict(preprocessed_image)[0].tolist()
          zipped = sorted(list(zip(classes, probs)), key=lambda x: x[1], reverse=True)
          image_class = [zipped[i][0] for i in range(len(zipped))]
          probability  = [zipped[i][1]*100 for i in range(len(zipped))]
          df = DataFrame(data={'image_class':image_class, 'probability(%)': probability})          
          print(df)
          return df
        
        
        # preprocess for the model input
        preprocessed_img = image_preprocessing(resized_img)


        # get the predictions
        dff = image_classification(preprocessed_img, model)
        c1 = dff['image_class']
        c2 = dff['probability(%)']
        sclass = c1[dff.index[0]] 
        probability = c2[dff.index[0]]
        sprob = probability
        
        # if the probability for 30zone is greater, show the image and warn
        if sclass == 'mixed':
            speed_limit = '30'
            self.speedimage.source = 'speed_logo.png'
        elif sclass == 'speed30zone':
            speed_limit = '30 Z'
            playsound('warning.mp3')
            self.speedimage.source = 'image30zone.png'
        else:
            speed_limit = '---' 
        
        
        
      
if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()

