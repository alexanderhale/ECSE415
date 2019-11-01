import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv

images = [cv2.imread(file) for file in glob.glob("../FinalProject/photos/preprocessed/train/*.jpg")]
faceBoxes = np.empty((len(images), 4), dtype=int)

for i, img in enumerate(images):
    faces, confidences = cv.detect_face(img)
    for face in faces:
        faceBoxes[i] = (face[0], face[1], face[2], face[3])
    print(faceBoxes[i])
    
    if i % 10 == 0:
        cv2.rectangle(img, (faceBoxes[i][0], faceBoxes[i][1]), (faceBoxes[i][2],faceBoxes[i][3]), (0,255,0), 2)
        # display output        
        plt.imshow(img)
        plt.show()

print(len(faceBoxes))
np.savetxt("../FinalProject/photos/preprocessed/train/faceBoxes.txt", faceBoxes)