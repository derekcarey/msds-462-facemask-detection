# Facemask detection web application using TensorFlow and Flask
 Flask application for facemask detection. For a detailed synopsis please the the following document:
 
 https://github.com/derk924/msds-462-facemask-detection/blob/main/Project%20Synopsis.md
 
 

# Instructions for execution: 

The github repo does not include the mask_detection_model folder due to filesize restrictions. A compressed version of the folder can be downloaded from the following S3 location:

https://derek-public-files.s3.amazonaws.com/mask_detection_model.zip

Once downloaded this folder must be extracted and placed within the VideoStreaming directory. 


The Flask server and application is started by executing the command 'python main.py' from the VideoStreaming directory. The application will run locally on the device network using and can be accessed using the following url:

http://{LOCAL IP ADDRESS}:5000

For example, when executed on a device with a local IP of 192.168.86.61, the web appication can be accessed on the local network by entering the  url:

http://192.168.86.61:5000


# Reperforming model training

The file MSDS_462_Mask_Detection_Model_Training.ipynb can be used for training a new model. The original model training was performed using the following Google Colab notebook:

https://colab.research.google.com/drive/1DbR6XFv_JJU-IkqFBzsJETeFiy_8xUY2?usp=sharing

The notebook assuming image contents are available within a Google Drive location at '/content/drive/My Drive/msds/462/mask-data/'. The dataset can be downloaded from kaggle's Face Mask Classification dataset created by Dhruv Makwana at the following url:

https://www.kaggle.com/dhruvmak/face-mask-detection

# Package Details

The appplication was configured in an environment running the following python packages. 

tensorflow 2.3.1 <br />
tensorflow-gpu 2.3.1 <br />
tensorflow-gpu-estimator 2.3.0 <br />
h5py 2.10.0 <br />
pandas 1.1.4 <br />
Flask 1.1.2 <br />
Pillow 8.0.1 <br />
Keras 2.4.3 <br />
matplotlib 3.3.3 <br />
requests 2.25.0 <br />
numpy 1.18.5  <br />
opencv-python 4.4.0.46 <br />

# CUDA Toolkit 11.1 Update 1 

Execution of the facemask detection Flask application may require installation of the NVIDIA CUDA Toolkit available at the current location:

https://developer.nvidia.com/cuda-downloads

# Questions? 

For questions or feedback please email derek.carey@outlook.com
