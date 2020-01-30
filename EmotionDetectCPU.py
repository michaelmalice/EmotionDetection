import sys
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
from datetime import datetime
try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading
try:
    from imutils.video.pivideostream import PiVideoStream
    from imutils.video.filevideostream import FileVideoStream
    import imutils
except:
    pass

LABELS = ["neutral", "happy", "sad", "surprise", "anger"]

def camThread():

    plugin = IEPlugin(device="CPU")
    plugin.add_cpu_extension("./lib/libcpu_extension.so")
    print("successful")
    model_xml = "./models/face-detection-retail-0004.xml"
    model_bin = "./models/face-detection-retail-0004.bin"
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    print("test")
    Exec_net = plugin.load(network=net)
    print("successful also")

    emotion_model_xml = "./models/emotions-recognition-retail-0003.xml"
    emotion_model_bin = "./models/emotions-recognition-retail-0003.bin"
    emotionNet = IENetwork(model=emotion_model_xml, weights=emotion_model_bin)
    emotion_input_blob = next(iter(net.inputs))
    emotion_exec_net = plugin.load(network=emotionNet)
    print("Successfully loaded emotion model")


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('frame')
    i = 0
    start = datetime.now()
    fps = 0
    while(i <= i):
        faces = []
        ret, frame = cap.read()
        if ret:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prepimg = cv2.resize(frame, (300, 300))
            prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            outputs = Exec_net.infer(inputs={input_blob: prepimg})
            for k, v in outputs.items():
                #print(str(k) + ' is key to value ' + str(v))
                #print(type(v))
                #print(v.flatten())
                #print(type(v.flatten()))

                j = 0
                while float(v.flatten()[j + 2]) > .9:
                    image_id = v.flatten()[j]
                    label = v.flatten()[j + 1]
                    conf = v.flatten()[j + 2]
                    boxTopLeftX = v.flatten()[j + 3] * frame.shape[1]
                    boxTopLefty = v.flatten()[j + 4] * frame.shape[0]
                    boxBottomRightX = v.flatten()[j + 5] * frame.shape[1]
                    boxBottomRightY = v.flatten()[j + 6] * frame.shape[0]
                    faces.append([conf, (int(boxTopLeftX), int(boxTopLefty)), (int(boxBottomRightX), int(boxBottomRightY))])
                    j = j + 7

                for face in faces:
                    prepimg = frame[face[1][1]:face[2][1], face[1][0]:face[2][0]]
                    #prepimg = cv2.cvtColor(prepimg, cv2.COLOR_RGB2BGR)
                    prepimg = cv2.resize(frame, (64, 64))
                    prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
                    prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
                    emotion_outputs = emotion_exec_net.infer(inputs={emotion_input_blob: prepimg})
                    for keys, values in emotion_outputs.items():
                        print(type(values))
                        print("emotion key is " + str(keys) + " and value is " + str(values.flatten()))
                        print(int(np.argmax(values.flatten())))
                        emotion = LABELS[int(np.argmax(values.flatten()))]
                        prob = float(values.flatten()[int(np.argmax(values.flatten()))]) * 100
                        print(emotion)
                    frame = cv2.rectangle(np.array(frame), face[1], face[2], (0, 0, 255), 1)
                    if prob > 60:
                        frame = cv2.putText(frame, emotion + "  confidence: " + "%.2f" % prob + "%", (face[1][0], face[1][1] - 2), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
                    #print(str(face[0]))

            frame = cv2.putText(frame, "FPS: " + "%.2f" % fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

            print("show image")
            cv2.imshow('frame', frame)
            #print("show frame")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i = i + 1
        if i > 30:
            fps = i/int((datetime.now() - start).total_seconds())
            print(fps)

    cap.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument('-cm','--modeofcamera',dest='mode_of_camera',type=int,default=0,help='Camera Mode. 0:=USB Camera, 1:=PiCamera (Default=0)')
parser.add_argument('-cn','--numberofcamera',dest='number_of_camera',type=int,default=0,help='USB camera number. (Default=0)')
parser.add_argument('-wd','--width',dest='camera_width',type=int,default=640,help='Width of the frames in the video stream. (Default=640)')
parser.add_argument('-ht','--height',dest='camera_height',type=int,default=480,help='Height of the frames in the video stream. (Default=480)')
parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
parser.add_argument('-vidfps','--fpsofvideo',dest='fps_of_video',type=int,default=30,help='FPS of Video. (Default=30)')
parser.add_argument('-fdmp','--facedetectionmodelpath',dest='fd_model_path',default='./models/face-detection-retail-0004',help='Face Detection model path. (xml and bin. Except extension.)')
parser.add_argument('-emmp','--emotionrecognitionmodelpath',dest='em_model_path',default='./models/emotions-recognition-retail-0003',help='Emotion Recognition model path. (xml and bin. Except extension.)')

args = parser.parse_args()
mode_of_camera = args.mode_of_camera
number_of_camera = args.number_of_camera
camera_width  = args.camera_width
camera_height = args.camera_height
number_of_ncs = args.number_of_ncs
vidfps = args.fps_of_video
fd_model_path = args.fd_model_path
em_model_path = args.em_model_path

camThread()
