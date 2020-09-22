import cv2
import numpy as np
import time as t
import winsound

print ("Package Imported")
folder = "C:/Users/Hilaire Yuma/Documents/Bois de boulogne/Cours420-A62_ProjetdeSynthese/Datasets"

# Information for the sound
frequency = 2500
duration = 1000

l = []
r = range(3)
for i in r:
    l.append("/Test" + str(i)+ ".png")

cap = cv2.VideoCapture(0)
for x in l:
    while True:
        success, img = cap.read()
        cv2.imshow("video", img)
        t.sleep(6)
        #cv2.imwrite(folder + x, img, [cv2.IMWRITE_JPEG_QUALITY,50])
        cv2.imwrite(folder + x, img)
        break

    # Lorsque ca sonne change de pause
    winsound.Beep(frequency, duration)
cv2.destroyAllWindows()
