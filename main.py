import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import io
from PIL import Image
from mangum import Mangum
# Initialize the Flask application
app = FastAPI()


model = YOLO('model/best.pt')

def findByModel(binary_image,threshold=0.5):

    image = np.array(Image.open(io.BytesIO(binary_image))) 
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    screenshot = img
    results = model(screenshot)[0]
    points = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(screenshot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            center_x = ((int(x2) - int(x1)) /2) + int(x1)
            center_y = ((int(y2) - int(y1)) /2) + int(y1)
            points.append((int(center_x), int(center_y)))
    return points

# route http posts to this method
@app.post("/predict")
async def predict(data: Request):
    data_b = await data.body()
    point = findByModel(data_b)
    res_json = {}
    if len(point) > 0:
        res_json = {"x": point[0][0], "y": point[0][1]}
    else:
        res_json = {"x": 0, "y": 0}
    return JSONResponse(res_json)


