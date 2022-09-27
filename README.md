# Transfer learning (Convolutional Neural Network )

## Overview

Detecting speed sign boards is important while driving. It helps to prevent accidents and fines. It is particulalry important when the rules change when entering into a new street. 30 Zone is one of the region in Germany for which driving rules change. If one can not notice the 30 Zone board at the very beginning of the street, it is possible she/he can have an accident.

This application detects 30 Zone boards. When a 30 Zone board is detected, it warns the driver and  shows a 30 Zone image.  


## Implementation of te Algorithm

It uses pre-trained model mobilnetv2. The maximum resolution mobilnetv2 accept is 224x224. The pickle file is generated in the folder "get_trained_model". 
The algorithm is run by the 'main.py' file. Below is the video that shows success of the method.

https://www.youtube.com/watch?v=yX-CYU37t1k



## Tech Stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)  ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23#ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=white)  	![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![kivy](https://img.shields.io/badge/kivy-%230C55A5.svg?style=for-the-badge&logo=kivy&logoColor=%white)
