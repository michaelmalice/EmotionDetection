import sys
import numpy as np
import cv2, io, time, argparse, re
import os
import platform
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


def camThread(device, number_of_camera, camera_width, camera_height, number_of_ncs, video, precision):
    if device == 'CPU':
        plugin = IEPlugin(device="CPU")
        if platform.system() == "Linux":
            plugin.add_cpu_extension('./lib/libcpu_extension.so')
        elif platform.system() == "Windows":
            print(os.getcwd())
            plugin.add_cpu_extension(os.getcwd() + '\\lib\\cpu_extension.dll')
        print("successfully loaded CPU plugin")
        if precision == "FP32":
            model_xml = "./models/FP32/face-detection-retail-0004.xml"
            model_bin = "./models/FP32/face-detection-retail-0004.bin"
            emotion_model_xml = "./models/FP32/emotions-recognition-retail-0003.xml"
            emotion_model_bin = "./models/FP32/emotions-recognition-retail-0003.bin"
        elif precision == "INT8":
            model_xml = "./models/INT8/face-detection-retail-0004.xml"
            model_bin = "./models/INT8/face-detection-retail-0004.bin"
            emotion_model_xml = "./models/INT8/emotions-recognition-retail-0003.xml"
            emotion_model_bin = "./models/INT8/emotions-recognition-retail-0003.bin"
    if device == "MYRIAD":
        plugin = IEPlugin(device="MYRIAD")
        print("Successfully loaded MYRIAD plugin")
        model_xml = "./models/FP16/face-detection-retail-0004.xml"
        model_bin = "./models/FP16/face-detection-retail-0004.bin"
        emotion_model_xml = "./models/FP16/emotions-recognition-retail-0003.xml"
        emotion_model_bin = "./models/FP16/emotions-recognition-retail-0003.bin"

    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    Exec_net = plugin.load(network=net)
    print("successfully loaded face model")
    emotionNet = IENetwork(model=emotion_model_xml, weights=emotion_model_bin)
    emotion_input_blob = next(iter(emotionNet.inputs))
    emotion_output_blob = next(iter(emotionNet.outputs))
    emotion_exec_net = plugin.load(network=emotionNet)
    print("Successfully loaded emotion model")

    if video == "":
        cap = cv2.VideoCapture(number_of_camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    else:
        cap = cv2.VideoCapture(video)

    cv2.namedWindow('frame')
    i = 0
    start = datetime.now()
    fps = 0
    while (cap.isOpened()):
        faces = []
        ret, frame = cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prepimg = cv2.resize(frame, (300, 300))
            prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            infer_request_handle = Exec_net.start_async(request_id=0, inputs={input_blob: prepimg})
            infer_status = infer_request_handle.wait()
            outputs = infer_request_handle.outputs[output_blob]
            outputs = outputs.flatten()
            j = 0
            while float(outputs[j + 2] > .9):
                image_id = outputs[j]
                label = outputs[j + 1]
                conf = outputs[j + 2]
                boxTopLeftX = outputs[j + 3] * frame.shape[1]
                boxTopLeftY = outputs[j + 4] * frame.shape[0]
                boxBottomRightX = outputs[j + 5] * frame.shape[1]
                boxBottomRightY = outputs[j + 6] * frame.shape[0]
                faces.append([conf, (int(boxTopLeftX), int(boxTopLeftY)), (int(boxBottomRightX), int(boxBottomRightY))])
                j = j + 7
            #Loop through faces and async infer, adding each handler to a dict. Then go through dict and draw on frame.
            emotion_results = []
            request_handler = None
            for index, face in enumerate(faces):
                prepimg = frame[face[1][1]:face[2][1], face[1][0]:face[2][0]]
                prepimg = cv2.resize(frame, (64, 64))
                prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
                prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
                if index > 0:
                    start_time = datetime.now()
                    print("Waiting")
                    emotion_outputs_status = request_handler[0].wait()
                    end_time = datetime.now()
                    print(end_time - start_time)
                    emotion_results.append((request_handler[0].outputs[emotion_output_blob].flatten(),
                                            request_handler[1]))
                request_handler = (emotion_exec_net.start_async(request_id=0, inputs={emotion_input_blob: prepimg}),
                                           face)
                print("started async infer")
            if request_handler:
                emotion_outputs_status = request_handler[0].wait()
                emotion_results.append(emotion_results.append((request_handler[0].outputs[emotion_output_blob].flatten(),
                                            request_handler[1])))
            for emotion_result in emotion_results:
                if emotion_result != None:
                    emotion_probs = emotion_result[0]
                    face = emotion_result[1]
                    emotion = LABELS[int(np.argmax(emotion_probs))]
                    prob = float(emotion_probs[int(np.argmax(emotion_probs))]) * 100
                    print(emotion)
                    neutral = emotion_probs[0]
                    happy = emotion_probs[1]
                    sad = emotion_probs[2]
                    surprise = emotion_probs[3]
                    anger = emotion_probs[4]
                    prob_scale_x = face[1][0] - 100

                    frame = cv2.rectangle(np.array(frame), face[1], face[2], (0, 0, 255), 1)
                    frame = cv2.rectangle(np.array(frame), (face[1][0] - 100, face[1][1]),
                                              (face[1][0], face[1][1] + 50),
                                              (0, 0, 255), cv2.FILLED)

                    frame = cv2.rectangle(np.array(frame), (prob_scale_x, face[1][1]),
                                              (face[1][0] + int(neutral * 100 - 100), face[1][1] + 10),
                                              (0, 255, 0), cv2.FILLED)
                    frame = cv2.putText(frame, "Neutral", (prob_scale_x + 60, face[1][1] + 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            .3, (0, 0, 0), 1)

                    frame = cv2.rectangle(np.array(frame), (prob_scale_x, face[1][1] + 10),
                                              (face[1][0] + int(happy * 100 - 100), face[1][1] + 20),
                                              (0, 255, 0), cv2.FILLED)
                    frame = cv2.putText(frame, "Happy", (prob_scale_x + 60, face[1][1] + 17), cv2.FONT_HERSHEY_SIMPLEX,
                                            .3, (0, 0, 0), 1)

                    frame = cv2.rectangle(np.array(frame), (prob_scale_x, face[1][1] + 20),
                                              (face[1][0] + int(sad * 100 - 100), face[1][1] + 30),
                                              (0, 255, 0), cv2.FILLED)
                    frame = cv2.putText(frame, "Sad", (prob_scale_x + 60, face[1][1] + 27), cv2.FONT_HERSHEY_SIMPLEX,
                                            .3, (0, 0, 0), 1)

                    frame = cv2.rectangle(np.array(frame), (prob_scale_x, face[1][1] + 30),
                                              (face[1][0] + int(surprise * 100 - 100), face[1][1] + 40),
                                              (0, 255, 0), cv2.FILLED)
                    frame = cv2.putText(frame, "Surprise", (prob_scale_x + 60, face[1][1] + 37), cv2.FONT_HERSHEY_SIMPLEX,
                                            .3, (0, 0, 0), 1)

                    frame = cv2.rectangle(np.array(frame), (prob_scale_x, face[1][1] + 40),
                                              (face[1][0] + int(anger * 100 - 100), face[1][1] + 50),
                                              (0, 255, 0), cv2.FILLED)
                    frame = cv2.putText(frame, "Anger", (prob_scale_x + 60, face[1][1] + 47), cv2.FONT_HERSHEY_SIMPLEX,
                                            .3, (0, 0, 0), 1)

                    if prob > 60:
                        frame = cv2.putText(frame, emotion + "  confidence: " + "%.2f" % prob + "%",
                                                (face[1][0], face[1][1] - 2), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                                                cv2.LINE_AA)

                        # print(str(face[0]))

            frame = cv2.putText(frame, "FPS: " + "%.2f" % fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

            print("show image")
            cv2.imshow('frame', frame)
        else:
            break
            # print("show frame")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i = i + 1
        if int((datetime.now() - start).total_seconds()) > 0:
            fps = i / int((datetime.now() - start).total_seconds())
            print(fps)
    print("Total time: " + str(datetime.now() - start))
    cap.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', dest='device', type=str, default='CPU',
                    help='Device to run inference on. Valid choices are CPU or MYRIAD. Default=CPU')
parser.add_argument('-p', '--precision', dest='precision', type=str, default='FP32',
                    help='Precision of model to be used. Options are FP32 and INT8. Default:FP32')
parser.add_argument('-cn', '--numberofcamera', dest='number_of_camera', type=int, default=0,
                    help='USB camera number. (Default=0)')
parser.add_argument('-vd', '--video', dest='video', type=str, default='',
                    help='File name of video file to test on. Default:blank')
parser.add_argument('-wd', '--width', dest='camera_width', type=int, default=640,
                    help='Width of the frames in the video stream. (Default=640)')
parser.add_argument('-ht', '--height', dest='camera_height', type=int, default=480,
                    help='Height of the frames in the video stream. (Default=480)')
parser.add_argument('-numncs', '--numberofncs', dest='number_of_ncs', type=int, default=0,
                    help='Number of NCS. (Default=0, unless device is MYRIAD, then Default=1)')


args = parser.parse_args()
device = args.device
precision = args.precision
number_of_camera = args.number_of_camera
video = args.video
camera_width = args.camera_width
camera_height = args.camera_height
number_of_ncs = args.number_of_ncs
if device == 'MYRIAD' and number_of_ncs < 1:
    number_of_ncs = 1


camThread(device, number_of_camera, camera_width, camera_height, number_of_ncs, video, precision)
