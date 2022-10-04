# Load packages
from fastapi import FastAPI, HTTPException, status
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from cv import predict_tree_count
import numpy as np
import cv2
import os
import datetime

# Initialize instance of FastAPI
app = FastAPI(title = "Tree Counting FastAPI")

# CORS Middleware
app.add_middleware(
        CORSMiddleware,
        allow_origins = ["*"],
        allow_methods = ["GET", "POST"],
        allow_headers = ["*"],
)

# Get output + header
def get_tree_count_response(contents):
    """
    Args: 
        contents: Reads file in ["jpg", "jpeg", "png"]

    Return:
        headers: response header
        out_file: output image file
    """
    arr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Predict tree count
    tree_count, imgOut = predict_tree_count(img)
    
    # Create output folder if not exists
    os.makedirs('output', exist_ok = True)

    # Save output to disk
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join('output', f'output_{now_str}.jpg') # Modified 20220914 - PNG to JPG
    cv2.imwrite(out_file, imgOut)

    # headers
    headers = {'tree_count': str(tree_count)}

    # return header and out_file
    return headers, out_file

# Index
@app.get("/")
def root():
    return {"message": "Tree Counting"}

@app.post("/predict/tree_count")
async def get_tree_count(file: UploadFile = File(...)):

    extension = file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")

    if not extension:
        raise HTTPException(status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail = "Image must be in jpg or png")

    contents = await file.read()
    headers, out_file = get_tree_count_response(contents)
    # arr = np.fromstring(contents, np.uint8)
    # img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    ## Predict tree count
    # tree_count, imgOut = predict_tree_count(img)

    ## Create output folder if not exists
    # os.makedirs('output', exist_ok = True)

    ## Save output to disk
    # now = datetime.datetime.now()
    # now_str = now.strftime('%Y%m%d_%H%M%S')
    # out_file = os.path.join('output', f'output_{now_str}.png')
    # cv2.imwrite(out_file, imgOut)
    
    ## headers
    # headers = {'tree_count': str(tree_count)}
    
    return headers
    # return FileResponse(out_file, headers = headers)

@app.post("/predict/tree_bbox")
async def get_tree_bbox(file: UploadFile = File(...)):

    extension = file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")

    if not extension:
        raise HTTPException(status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail = "Image must be in jpg or png")

    contents = await file.read()
    headers, out_file = get_tree_count_response(contents)

    return FileResponse(out_file, headers = headers)
