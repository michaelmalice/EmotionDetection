import sys
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
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


    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame')
    i = 0
    while(i == i):
        faces = []
        ret, frame = cap.read()
        if ret:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prepimg = cv2.resize(frame, (300, 300))
            prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            outputs = Exec_net.infer(inputs={input_blob: prepimg})
            for k, v in outputs.items():
                print(str(k) + ' is key to value ' + str(v))
                print(type(v))
                print(v.flatten())
                print(type(v.flatten()))

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
                    frame = cv2.rectangle(np.array(frame), face[1], face[2], (0, 0, 255), 2)
                    frame = cv2.putText(frame, "confidence: " + str(face[0]), (face[1][0], face[1][1] - 2), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
                    print(str(face[0]))
            print("show image")
            cv2.imshow('frame', frame)
            #print("show frame")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i = i + 1

    cap.release()
    cv2.destroyAllWindows()





camThread()


'''                for i in range(7):
                    print(v.flatten()[i])
                image_id = v.flatten()[0]
                label = v.flatten()[1]
                conf = v.flatten()[2]
                print("confidence is " + str(float(conf)))
                boxTopLeftX = v.flatten()[3] * frame.shape[1]
                boxTopLeftY = v.flatten()[4] * frame.shape[0]
                boxBottomRightX = v.flatten()[5] * frame.shape[1]
                boxBottomRightY = v.flatten()[6] * frame.shape[0]
                for i in range(7, 50):
                    if i%7 + 2 == 2:
                        print("confidence is " + str(float(v.flatten()[i + 2])))
                    #print(v.flatten()[i])
                print('frame is type ' + str(type(frame)))

            #res = outputs[out_blob]
            #res
            #print(outputs[0])
            #print(res)'''