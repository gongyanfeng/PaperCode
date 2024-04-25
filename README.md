# PaperCode
The code corresponds to the paper "Two-stage ***"

# 1. Train the model
## 1.1 train the first stage using following script
python train first_stage.py
## 1.2 train the second stage using following script
python train second_stage.py

# 2. Test the model
After training is completed， you can use the following script to test an image:
python predict.py --img image_path
例如：
python predict.py --img ./images/7_5.jpg
   
