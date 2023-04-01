# usbcamera-to-videofile.py
# 4/1/23
# save each frame to an mp4V file. 

import cv2 as cv
import numpy as np 
import torch
import seaborn as sn

# object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)

# open the file (or the video usb camera)
usbcamera = 0
vid = cv.VideoCapture(usbcamera)

# set new size of video capture (WxH combination doesn't always work)
ret = vid.set(cv.CAP_PROP_FRAME_WIDTH,1280)
ret = vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# read and display one frame
ret, frame = vid.read()

# if we're resaving this video, create a VideoWriter 
videoformat = cv.VideoWriter_fourcc(*'mp4v')  # .mp4
# videoformat = cv.VideoWriter_fourcc(*'XVID')  # .mp4
# other formats for Mac OSX, note *'MJPG' is the same as ('M','J','P','G')
# videoformat = cv.VideoWriter_fourcc(*'MJPG')   # .mp4 (does not play in quicktime, play in VLC)
# videoformat = cv.VideoWriter_fourcc(*'DIVX')   # .avi 
# videoformat = cv.VideoWriter_fourcc(*'X264')   # .mkv 
# fps = int(vid.get(cv.CAP_PROP_FPS))  # fps 
fps = 15  # 15fps seems to be the normal speed 
width  = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))  # float `width`
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
framesize = (width, height)
filenameOut = 'april1-testing-mp4v-15fps-1280-720.mp4'
videoOutput =  cv.VideoWriter(filenameOut, videoformat, fps, framesize)

# while file (or camera) is open, do the following loop
while vid.isOpened():
    # read frame of video file
    ret, frame = vid.read()

    # is frame valid? check the return value
    if not ret:
        print("End of file... exiting")
        break 
    
    # do object detection
    results = model(frame)
    df = results.pandas().xyxy[0] 
    if not df.empty:
        objdetected = df['name'].values[0]
        # check if bird detected 
        if objdetected == 'bird':
            print(f'object detected: {objdetected}')
            # write the frame
            videoOutput.write(frame)

            # display in window (note: there will be no sound)
            cv.imshow("My Video", frame)
            
    # wait for keypress 'q' to exit
    if cv.waitKey(1) == ord('q'):
        # close video output file
        videoOutput.release()
        # close video file 
        vid.release() 
        # close any window created
        cv.destroyAllWindows()
        break 

