# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import io
# import numpy as np

# from utils.analyze import process_image_and_calculate_creatinine

# app = FastAPI()

# # Allow frontend (Flutter) to access backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/analyze/")
# async def analyze_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")
#     image_np = np.array(image)

#     result = process_image_and_calculate_creatinine(image_np)
#     return {"creatinine_value": result}



# # NEW FILE CODE

# from utils.auto_crop import auto_crop_and_resize
# from utils.creatinine_calc import get_creatinine_value
# import os

# image_path = 'inputs/sample_image.jpg'
# output_folder = 'outputs'

# # Step 1: Try auto crop
# try:
#     cropped_path = auto_crop_and_resize(image_path)
#     print(f"[AutoCrop] Cropped Image saved at: {cropped_path}")
# except Exception as e:
#     print(f"[AutoCrop] Failed: {e}")
#     cropped_path = image_path  # fallback to original

# # Step 2: Calculate Creatinine Value
# try:
#     value = get_creatinine_value(cropped_path, output_folder)
#     print(f"\n[Result] Final Creatinine Value: {value:.2f} mg/dL")
# except Exception as e:
#     print(f"[Error] Creatinine Calculation Failed: {e}")





# # NEW FILE 2

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import numpy as np
# import os
# import shutil
# from tempfile import NamedTemporaryFile

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Update for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------- Utility Functions ----------

# def get_correction_factors(bg_color, std_values=(200, 200, 200)):
#     return [std / c if c > 0 else 1 for c, std in zip(bg_color, std_values)]

# def apply_correction(color, factors):
#     corrected = [min(255, max(0, round(c * f))) for c, f in zip(color, factors)]
#     return np.array(corrected, dtype=np.uint8)

# def process_image(image_path, output_folder="corrected_images"):
#     image = cv2.imread(image_path)

#     if image is None or image.shape != (300, 1000, 3):
#         raise ValueError("Uploaded image must be exactly 1000x300 pixels in size.")

#     # Define regions
#     region_1 = image[:, 0:166]           # Left background
#     region_5 = image[:, 833:1000]        # Right background
#     region_3 = image[:, 420:580]         # Foreground (tip)

#     # Compute average colors
#     bg_mean_1 = region_1.mean(axis=(0, 1))
#     bg_mean_5 = region_5.mean(axis=(0, 1))
#     bg_mean = (bg_mean_1 + bg_mean_5) / 2
#     tip_mean = region_3.mean(axis=(0, 1))

#     # Correct tip color
#     correction_factors = get_correction_factors(bg_mean, std_values=(200, 200, 200))
#     corrected_color = apply_correction(tip_mean, correction_factors)

#     # Save corrected patch
#     os.makedirs(output_folder, exist_ok=True)
#     corrected_img = np.full((200, 100, 3), corrected_color, dtype=np.uint8)
#     filename = os.path.basename(image_path).replace(".jpg", "_corrected.jpg")
#     save_path = os.path.join(output_folder, filename)
#     cv2.imwrite(save_path, corrected_img)

#     # Calculate creatinine value
#     R = corrected_color[2]  # BGR → R
#     creatinine_value = round((208 - R) / 11.6, 2)

#     return {
#         "creatinine_value": creatinine_value,
#         "corrected_rgb": {
#             "R": int(corrected_color[2]),
#             "G": int(corrected_color[1]),
#             "B": int(corrected_color[0])
#         },
#         "corrected_image_path": save_path
#     }

# # ---------- API Endpoint ----------

# @app.post("/analyze-creatinine")
# async def analyze_creatinine(file: UploadFile = File(...)):
#     try:
#         # Save uploaded file temporarily
#         with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#             shutil.copyfileobj(file.file, tmp)
#             tmp_path = tmp.name

#         # Process and get result
#         result = process_image(tmp_path)

#         # Remove temp image
#         os.remove(tmp_path)

#         return JSONResponse(content={
#             "message": "Success",
#             "creatinine_mg_dL": result["creatinine_value"],
#             "corrected_rgb": result["corrected_rgb"],
#             "corrected_image": result["corrected_image_path"]
#         })

#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})





from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from tempfile import NamedTemporaryFile
import traceback

# Importing your modules
from utils.auto_crop import auto_crop
from utils.creatinine_calc import get_creatinine_value

app = FastAPI()

# Allow all origins — change this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- CONFIG ----
OUTPUT_FOLDER = "corrected_images"

# # ---- API Endpoint ----

# @app.post("/analyze-creatinine")
# async def analyze_creatinine(file: UploadFile = File(...)):
#     try:
#         # Step 1: Save the uploaded file temporarily
#         with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#             shutil.copyfileobj(file.file, tmp)
#             tmp_path = tmp.name

#         # Step 2: Auto-crop the image
#         cropped_path = auto_crop(tmp_path)

#         # Step 3: Get creatinine value and corrected patch
#         creatinine_value = get_creatinine_value(cropped_path, output_folder=OUTPUT_FOLDER)

#         # Step 4: Cleanup temporary files
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

#         return JSONResponse(content={
#             "message": "Success",
#             "creatinine_mg_dL": creatinine_value,
#             "corrected_image": os.path.join(OUTPUT_FOLDER, os.path.basename(cropped_path).replace(".jpg", "_corrected.jpg"))
#         })

#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})




@app.post("/analyze-creatinine")
async def analyze_creatinine(file: UploadFile = File(...)):
    try:
        print("🔵 Received file:", file.filename)

        # Step 1: Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        print("📂 Saved temp file at:", tmp_path)

        # Step 2: Auto-crop the image
        cropped_path = auto_crop(tmp_path)
        print("✂️ Cropped image saved at:", cropped_path)

        # Step 3: Get creatinine value and corrected patch
        creatinine_value = get_creatinine_value(cropped_path, output_folder=OUTPUT_FOLDER)
        print("🧪 Creatinine Value:", creatinine_value)

        # Optional: path of corrected image
        corrected_filename = os.path.basename(cropped_path).replace(".jpg", "_corrected.jpg")
        corrected_path = os.path.join(OUTPUT_FOLDER, corrected_filename)
        print("✅ Final corrected image path:", corrected_path)

        # Step 4: Cleanup temp
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return JSONResponse(content={
            "message": "Success",
            "creatinine_value": round(creatinine_value, 2),  # renamed for frontend
            "corrected_image": corrected_filename  # use just filename instead of full path
        })

    except Exception as e:
        print("🔥 Error occurred:")
        traceback.print_exc()
        return JSONResponse(status_code=400, content={
            "error": str(e)
        })




