import numpy as np
import cv2
import datetime
cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'MPEG')
file=cv2.VideoWriter('file1.avi',fourcc,20,(640,480))
while(cap.isOpened()):
    ret,frame=cap.read()
    #print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    dt=str(datetime.datetime.now())
    f=cv2.FONT_HERSHEY_SIMPLEX
    frame=cv2.putText(frame,dt,(5,20),f,0.5,(255,255,255),1,cv2.LINE_AA)
    file.write(frame)
    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
file.release()
cv2.destroyAllWindows()
