Use Viola Jones algorithm to do face detection. Training image set is FDDB.
## How to use
 - Set up FDDB traning folder
    - Put all training images .jpg files in path “./FDDB/originalPics/”
    - Put all ground truth txt files in path “./FDDB/FDDB-folds/”
    - Put all testing images .jpg files in test path
 - Run python3 ViolaJones.py
 - The json file that contains bounding box in test image set is created in result folder
