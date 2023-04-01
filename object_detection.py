import torch
import seaborn as sn


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)

# Images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
img = './unknown/Mourning-Dove-2.jpg'
# img = "./unknown/bluebird-480px.jpg"

# Inference
results = model(img)

# print(results)
# Results
# results.print()
results.save()  # or .show()
# print(results.xyxy[0])  # img1 predictions (tensor)
# print(results.pandas().xyxy[0])  # img1 predictions (pandas)
# print(results.pandas().xyxy[0])  # img1 predictions (pandas)

df = results.pandas().xyxy[0] 
objdetected = df['name'].values[0]

print(f'object detected by YOLO is a: {objdetected}')