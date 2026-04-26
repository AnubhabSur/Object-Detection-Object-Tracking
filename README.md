The object detection and object tracking works on yolo8 model versions(s,n)
First the video is uploaded then it is broken into single frames then the detection of object is done.
The object tracking follows the same work module as the object detection but in addition it uses IoU(Intersection Over Union) to detect the similar object.
After IoU the same object is tracked through out the frames until it goes out the frame.
