# msds-462-facemask-detection
 Flask application for facemask detection

# Instructions for execution: 

The github repo does not include the mask_detection_model folder due to filesize restrictions. A compressed version of the folder can be downloaded from the following S3 location:

https://derek-public-files.s3.amazonaws.com/mask_detection_model.zip

The Flask server and application is started by executing the command 'python main.py' from the VideoStreaming directory. The application will run locally on the device network using and can be accessed using the following url:

http://{LOCAL IP ADDRESS}:50000

For example, when executed on a device with a local IP of 192.168.86.61, the web appication can be accessed on the local network by entering the  url:

http://192.168.86.61

Once downloaded this folder must be extracted and placed within the VideoStreaming directory. 

# Package Details

The appplication was configured in an environment running the following python packages. 



Package                  Version
------------------------ -------------------
absl-py                  0.11.0
argon2-cffi              20.1.0
astor                    0.8.1
astunparse               1.6.3
async-generator          1.10
attrs                    20.3.0
backcall                 0.2.0
bleach                   3.2.1
blinker                  1.4
brotlipy                 0.7.0
cached-property          1.5.2
cachetools               4.1.1
certifi                  2020.11.8
cffi                     1.14.4
chardet                  3.0.4
click                    7.1.2
colorama                 0.4.4
cryptography             3.2.1
cycler                   0.10.0
decorator                4.4.2
defusedxml               0.6.0
entrypoints              0.3
Flask                    1.1.2
gast                     0.3.3
google-auth              1.23.0
google-auth-oauthlib     0.4.2
google-pasta             0.2.0
grpcio                   1.34.0
h5py                     2.10.0
idna                     2.10
importlib-metadata       3.1.1
ipykernel                5.3.4
ipython                  7.19.0
ipython-genutils         0.2.0
ipywidgets               7.5.1
itsdangerous             1.1.0
jedi                     0.17.2
Jinja2                   2.11.2
jsonschema               3.2.0
jupyter                  1.0.0
jupyter-client           6.1.7
jupyter-console          6.2.0
jupyter-core             4.7.0
jupyterlab-pygments      0.1.2
Keras                    2.4.3
Keras-Applications       1.0.8
Keras-Preprocessing      1.1.2
kiwisolver               1.3.1
Markdown                 3.3.3
MarkupSafe               1.1.1
matplotlib               3.3.3
mistune                  0.8.4
mkl-fft                  1.2.0
mkl-random               1.1.1
mkl-service              2.3.0
nbclient                 0.5.1
nbconvert                6.0.7
nbformat                 5.0.8
nest-asyncio             1.4.3
notebook                 6.1.5
numpy                    1.18.5
oauthlib                 3.1.0
opencv-python            4.4.0.46
opt-einsum               3.3.0
packaging                20.7
pandas                   1.1.4
pandocfilters            1.4.3
parso                    0.7.1
pickleshare              0.7.5
Pillow                   8.0.1
pip                      20.3.1
prometheus-client        0.9.0
prompt-toolkit           3.0.8
protobuf                 3.14.0
pyasn1                   0.4.8
pyasn1-modules           0.2.8
pycparser                2.20
Pygments                 2.7.2
PyJWT                    1.7.1
pyOpenSSL                20.0.0
pyparsing                2.4.7
pyreadline               2.1
pyrsistent               0.17.3
PySocks                  1.7.1
python-dateutil          2.8.1
pytz                     2020.4
pywin32                  300
pywinpty                 0.5.7
PyYAML                   5.3.1
pyzmq                    20.0.0
qtconsole                5.0.1
QtPy                     1.9.0
requests                 2.25.0
requests-oauthlib        1.3.0
rsa                      4.6
scipy                    1.5.4
Send2Trash               1.5.0
setuptools               50.3.2.post20201201
six                      1.15.0
tensorboard              2.3.0
tensorboard-plugin-wit   1.7.0
tensorflow               2.3.1
tensorflow-estimator     2.3.0
tensorflow-gpu           2.3.1
tensorflow-gpu-estimator 2.3.0
termcolor                1.1.0
terminado                0.9.1
testpath                 0.4.4
tornado                  6.1
traitlets                5.0.5
urllib3                  1.26.2
wcwidth                  0.2.5
webencodings             0.5.1
Werkzeug                 1.0.1
wheel                    0.36.0
widgetsnbextension       3.5.1
win-inet-pton            1.1.0
wincertstore             0.2
wrapt                    1.12.1
zipp                     3.4.0
