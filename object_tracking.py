import cv2
import random
from ultralytics import YOLO

def get_colors(cls_num):
  random.seed(cls_num)
  return tuple(random.randint(0, 255) for _ in range(3))

video_path= "D:\\Deep_sort\\6.mp4"
videoCap = cv2.VideoCapture(video_path)

model= YOLO('yolov8s.pt')

width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(videoCap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
output_video_path = 'D:\\Deep_sort\\tracked_output2.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not out.isOpened():
  print("Error: Could not open VideoWriter")
  exit()
  
frame_count = 0

while True:
  ret, frame = videoCap.read()

  if not ret:
    break

  results = model.track(frame, stream=True, device="cuda")

  for result in results:
    class_names = result.names
    
    if result.boxes.id is not None:
        boxes= result.boxes.xyxy.cpu()
        track_ids= result.boxes.id.int().cpu().tolist()
        clss= result.boxes.cls.cpu().tolist()
        confs= result.boxes.conf.cpu().tolist()

        for box, track_ids, cls, conf in zip(boxes,
                                             track_ids, 
                                             clss, 
                                             confs):
            x1, y1, x2, y2= map(int, box)
            cls = int(cls)
            class_name = class_names[cls]
            colour =  get_colors(cls)
        
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            
            label= f"ID:{track_ids} {class_name} {conf:.2f}"

            cv2.putText(frame, label,(x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6, colour, 2)

  out.write(frame)

  frame_count += 1
  if frame_count% 30== 0:
    print(f"Processed {frame_count} frames")
    
videoCap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video processing complete!")
print(f"Total frames processed: {frame_count}")
print(f"Output saved to: {output_video_path}")