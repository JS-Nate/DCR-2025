"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import pandas as pd
import array
import datetime
import time
import csv
Operator_name = input("Please Enter your name Before start the Scenario:")
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
Operator_name = input("Please Enter your name Before start the Scenario:")
Clmn_Hdr = ['Hr', 'Mn', 'Sc', 'Pupil']
results={"Eye Detection":[], "Time":[]}
dtTm_now = datetime.datetime.now()
dt = dtTm_now.strftime("%Y-%m-%d_%H_%M_%S")
EyTrc_Fl = dt + '_Ey.csv'

def LgFl_Crt(dt2):
    EyTrc_Fl = dt2 + '_Ey.csv'
    print (EyTrc_Fl)
    f = open(EyTrc_Fl, 'a+')    # Open to add to it
    writer = csv.writer(f, delimiter = ",")
    writer.writerow(Clmn_Hdr)
    import os
    #cmd = "sudo chown pi: /home/pi/EyDt/" + EyTrc_Fl
    #os.system(cmd)
    f.close

try:
    fl = open(EyTrc_Fl)
except:
    LgFl_Crt(dt)

EyTrc = array.array('i', [1, 2, 3, 4])


while True:

    dtTm_now = datetime.datetime.now()
    EyTrc[0] = dtTm_now.hour
    EyTrc[1] = dtTm_now.minute
    EyTrc[2] = dtTm_now.second
    # dt1 = dtTm_now.strftime("%Y%m%d")



    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""
    EyTrc[3] = 0

    if gaze.is_blinking():
        text = "Blinking"
        EyTrc[3] = 1
    elif gaze.is_right():
        text = "Looking right"
        EyTrc[3] = 2
    elif gaze.is_left():
        text = "Looking left"
        EyTrc[3] = 3
    elif gaze.is_center():
        text = "Looking center"
        EyTrc[3] = 4


    if text == "Blinking" or  text == "Looking right" or text == "Looking left" or text == "Looking center" :
        results["Eye Detection"].append(text)
        results["Time"].append(dtTm_now)
        print(text)
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Eye Tracking Example", frame)

    time.sleep(1/2)

    if cv2.waitKey(1) == 27:
        break
results=  pd.DataFrame(results)

results.to_csv(f"Results\\EyeTracking-{Operator_name}-{dt}.csv")
webcam.release()
cv2.destroyAllWindows()
