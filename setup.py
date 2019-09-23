from setuptools import setup

setup(
   name='yolo-v2',
   version='1.0',
   description='Bucket Tracking',
   author='Hooman Shariati',
   author_email='hooman@motionmetrics.com',
   packages=['yolo', 'utils', 'rnn', 'fm_frame_selection', 'data_gen'],  #same as name
   install_requires=[
        "Keras==2.2.4",
        "matplotlib==3.0.2",
        "numpy==1.15.4",
        "opencv-python==3.4.5.20",
        "Pillow==5.4.1",
        "tensorflow==1.12.2"
    ], #external packages as dependencies
)
