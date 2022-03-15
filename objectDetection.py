#import libraries
import cv2
import numpy as np

#loading weights and config files
#yolov3 615 
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes = []

#names converted into a list
with open('coco.names','r') as f:
    classes = f.read().splitlines()
    
    
# 0 in VideoCature function is used for webcam
# if given 'test.mp4' in VideoCapture then it can be used for Video Files
cap = cv2.VideoCapture(0)
#img = cv2.imread('image.jpg')

#while loop used for video if not remove it and its indentation
while True:
    _, img = cap.read()
    #capturing height and weight of image
    height,width,_ =img.shape

    #blob function used
    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB = True,crop = False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOuput = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
# get information from each object
    for ouput in layerOuput:
        for detect in ouput:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detect[0]*width)
                center_y = int(detect[1]*height)
                w = int(detect[2]*width)
                h = int(detect[3]*height)
            
                x = int(center_x - w/2)
                y = int(center_y - h/2)
            
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
            
    print(len(boxes))       
    # supreess lower value boxes
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4) 

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))

    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img,(x,y),(x+w , y+h),color,2)
        cv2.putText(img,label+" "+confidence,(x,y+20),font,2,(255,255,255),2)
    
        
            
            
        


    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.realease()
cv2.destroyAllWindows()


