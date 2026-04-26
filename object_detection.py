import cv2
import random
from ultralytics import YOLO

model= YOLO('yolov8s.pt')

def get_colors(cls_num):
  random.seed(cls_num)
  return tuple(random.randint(0, 255) for _ in range(3))

video_path= "D:\\Deep_sort\\1.mp4"
videoCap = cv2.VideoCapture(video_path)

width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(videoCap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
output_video_path = 'D:\\Deep_sort\\tracked_output4.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while True:
  ret, frame = videoCap.read()

  if not ret:
    break

  results = model.track(frame, stream=True, device="cuda")

  for result in results:
    class_names = result.names

    for box in result.boxes:
      if box.conf[0] > 0.4:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])
        class_name = class_names[cls]

        conf = float(box.conf[0])

        colour =  get_colors(cls)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        cv2.putText(frame, f"{class_name} {conf:.2f}",
        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, colour, 2)

  out.write(frame)

  frame_count += 1

videoCap.release()
out.release()
print(f"Processed video saved to {output_video_path}")