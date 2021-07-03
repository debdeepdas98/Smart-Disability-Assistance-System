import cv2
import numpy as np
import serial #connect Rasp. Pi to Arduino

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1) # begin sr.comm, 115200 baudrate used in arduino
    ser.flush() # clear old buffer
    # while True:
        # if ser.in_waiting > 0:
            # line = ser.readline().decode('utf-8').rstrip()
            # print(line)

# load YOLO algo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # read pre-trained model and config file
classes = []

# list create of coco names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames() # list layers of net
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()] # list layers

cap = cv2.VideoCapture(0) #cap-capture
while True:
    _, frame = cap.read() # returns frame
    height, width, channel = frame.shape # info of frame
    # creating blob 
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, False) # reshape, fx(scale fact, size, colour conversion, swap bgr<->rgb, no crop)

    # print blob
    # for b in blob:
    #    for n,im in enumerate(b):
     #       cv2.imshow(str(n),im)

    # giving the blob as a input to the yolo
    net.setInput(blob)
    # stroring result in outs
    outs = net.forward(output_layers) # func -outcomes of corresponding o/p layers, outs is md-array
    boxes = []
    confidences = []
    class_ids = []

    # analyzing the output
    for out in outs:
        for detection in out:
            scores = detection[5:] # first 5 contains centre-coordinates, disc - do not declare it beforehand, not user-created array
            class_id = np.argmax(scores) # max argument, np for md-array
            confidence = scores[class_id]
            if confidence > 0.6:  # updated on July'20
                # Object Detected
                # object center coordinae
                # adjust blob -> original image
                center_x = int(detection[0] * width) # for original
                center_y = int(detection[1] * height) 
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # rectangle coordinates
                # to create box
                x = int(center_x-w /2)
                y = int(center_y-h/2)
                boxes.append([x, y, w, h]) # actual coordinates
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # creating the rectangle and putting label in the objects
    num_obj_detec = len(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # decides final label based on more confidence, for eg monitor and tv
    for i in range(num_obj_detec):
        if i in indexes:
            x, y, w, h=boxes[i]
            label=classes[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # final output image
    # cv2.imshow("Output", frame)
    # key = cv2.waitKey(50)
    #if key == ord('q'):
    #    break


cap.release()
# cv2.destroyAllWindows()
