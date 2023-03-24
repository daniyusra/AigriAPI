from io import BytesIO
from PIL import Image
import numpy as np
from .models.disease_tracker import predict

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def get_prediction(image: Image.Image):    
    img = np.asarray(image.resize((256, 256)))[..., :3]
    #image = np.expand_dims(image, 0)
    img = img / (255.0) 
    img = img.astype(np.float32)
    
    #result = decode_predictions(model.predict(image), 2)[0]  TODO after model
    #for time being, use dummy data

    result = predict(img)

    #result = [[0, "Apple Disease", 0.5732], [1, "Covid Disease", 0.312], [2, "Fukuoka Disease", 0.006]]  
    response = []

    for i, res in enumerate(result): #returns top 2 of prediction results
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"        
        response.append(resp)    
    
    return response