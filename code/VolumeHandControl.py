import cv2
import time
from cv2 import FILLED
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volPer = 0
volBar = 400


cap = cv2.VideoCapture(0)
pTime=0

detector = htm.handDetector(detectionCon=0.7)

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        length, img, lineInfo = detector.findDistance(4, 8, img) 
        print(length)
        # print(length)

        # hand range 15 - 100
        # volume range -63.5 - 0

        # vol = np.interp(length,[15 ,120], [minVol, maxVol])
        volBar = np.interp(length,[15 ,120], [400, 150])
        volPer = np.interp(length,[15 ,120], [0, 100])
        volume.SetMasterVolumeLevelScalar(volPer/100, None)

        smoothness = 5
        volPer = smoothness * round(volPer/smoothness)

        if length<20:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
    
    cv2.rectangle(img, (50,150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'VOLUME: {int(volPer)} %', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText(img, f'FPS: {int(volPer)} %', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('img', img)
    cv2.waitKey(1)
