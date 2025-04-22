import cv2
import numpy as np
import os

def get_correction_factors(bg_color, std_values=(200, 200, 200)):
    """Compute correction factors to bring background close to a neutral reference color."""
    return [std / c if c > 0 else 1 for c, std in zip(bg_color, std_values)]  # Avoid division by zero

def apply_correction(color, factors):
    """Apply channel-wise correction using the given factors."""
    corrected = [min(255, max(0, round(c * f))) for c, f in zip(color, factors)]
    return np.array(corrected, dtype=np.uint8)


def get_creatinine_value(image_path, output_folder):
    image = cv2.imread(image_path)

    if image is None:
        raise Exception(f"Could not load image at path: {image_path}")

    if image.shape[1] != 1000 or image.shape[0] != 300:
        raise Exception("Expected image size: 1000x300.")

    # --- Define background and foreground regions ---
    region_1 = image[:, 0:166]            # Left background
    region_5 = image[:, 833:1000]         # Right background
    region_3 = image[:, 420:580]          # Tip region

    # --- Compute mean colors ---
    bg_mean_1 = region_1.mean(axis=(0, 1))
    bg_mean_5 = region_5.mean(axis=(0, 1))
    bg_mean = (bg_mean_1 + bg_mean_5) / 2
    tip_mean = region_3.mean(axis=(0, 1))

    # --- Calculate correction factors ---
    correction_factors = get_correction_factors(bg_mean, std_values=(200, 200, 200))
    corrected_tip_color = apply_correction(tip_mean, correction_factors)

    # --- Apply color correction to entire image ---
    corrected_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            original_pixel = image[y, x]
            corrected_pixel = apply_correction(original_pixel, correction_factors)
            corrected_image[y, x] = corrected_pixel

    # --- Save the corrected full image ---
    filename = os.path.basename(image_path).replace(".jpg", "_corrected.jpg").replace(".png", "_corrected.png")
    save_path = os.path.join(output_folder, filename)
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(save_path, corrected_image)

    # --- Calculate creatinine value ---
    R = corrected_tip_color[2]  # OpenCV = BGR
    G = corrected_tip_color[1]
    creatinine_value = round((209 - R) / 11.6, 2)
    # R = -11.6*x + 209

    # --- Output info ---
    print(f"[RGB] Corrected Tip Color: R={R}, G={G}, B={corrected_tip_color[0]}")
    print(f"[Save] Full Corrected Image saved at: {save_path}")
    print(f"[Creatinine] Value = {creatinine_value} mg/dL")

    return creatinine_value







# # CODE 1

# def get_creatinine_value(image_path, output_folder):
#     image = cv2.imread(image_path)
    
#     if image is None:
#         raise Exception(f"Could not load image at path: {image_path}")

#     if image.shape[1] != 1000 or image.shape[0] != 300:
#         raise Exception("Expected image size: 1000x300.")

#     # --- Define regions based on scaled pixel boundaries for 1000x300 image ---
#     region_1 = image[:, 0:166]           # Region 1: Left background
#     region_5 = image[:, 833:1000]        # Region 5: Right background
#     region_3 = image[:, 420:580]         # Region 3: Liquid (foreground)

#     # --- Calculate mean color values ---
#     bg_mean_1 = region_1.mean(axis=(0, 1))
#     bg_mean_5 = region_5.mean(axis=(0, 1))
#     bg_mean = (bg_mean_1 + bg_mean_5) / 2

#     tip_mean = region_3.mean(axis=(0, 1))

#     # --- Apply correction ---
#     correction_factors = get_correction_factors(bg_mean, std_values=(200, 200, 200))
#     corrected_color = apply_correction(tip_mean, correction_factors)


#     # --- Crop the original tip (region_3) ---
#     cropped_tip = region_3.copy()

#     # --- Apply correction to each pixel in the cropped tip ---
#     corrected_cropped = np.zeros_like(cropped_tip)
#     for y in range(cropped_tip.shape[0]):
#         for x in range(cropped_tip.shape[1]):
#             original_pixel = cropped_tip[y, x]
#             corrected_pixel = apply_correction(original_pixel, correction_factors)
#             corrected_cropped[y, x] = corrected_pixel

#     # --- Save the corrected cropped region ---
#     filename = os.path.basename(image_path).replace(".jpg", "_corrected.jpg")
#     save_path = os.path.join(output_folder, filename)
#     os.makedirs(output_folder, exist_ok=True)
#     cv2.imwrite(save_path, corrected_cropped)



#     # # --- Save corrected color image patch ---
#     # corrected_img = np.full((200, 100, 3), corrected_color, dtype=np.uint8)
#     # filename = os.path.basename(image_path).replace(".jpg", "_corrected.jpg")
#     # save_path = os.path.join(output_folder, filename)
#     # os.makedirs(output_folder, exist_ok=True)
#     # cv2.imwrite(save_path, corrected_img)


#     # --- Calculate creatinine value ---
#     R = corrected_color[2]  # OpenCV uses BGR, so R = index 2
#     G = corrected_color[1]
#     # creatinine_value = round((208 - R) / 11.6, 2)
#     creatinine_value = round((213 - G) / 7.89, 2)


#     print(f"[RGB] Corrected Tip Color: R={R}, G={corrected_color[1]}, B={corrected_color[0]}")
#     print(f"[Save] Corrected Patch saved at: {save_path}")
#     print(f"[Creatinine] Value = {creatinine_value} mg/dL")

#     return creatinine_value






# CODE 2

# import cv2
# import numpy as np
# import os

# def get_creatinine_value(image_path, output_folder):
#     image = cv2.imread(image_path)
#     if image.shape[1] != 1000 or image.shape[0] != 300:
#         raise Exception("Expected image size: 1000x300.")

#     region_1 = image[50:250, 50:150]
#     region_5 = image[50:250, 850:950]
#     region_3 = image[50:250, 450:550]

#     bg_mean = (region_1.mean(axis=(0,1)) + region_5.mean(axis=(0,1))) / 2
#     tip_mean = region_3.mean(axis=(0,1))

#     corrected = np.clip((tip_mean - bg_mean) + [128,128,128], 0, 255).astype(np.uint8)

#     corrected_img = np.full((200, 100, 3), corrected, dtype=np.uint8)
#     filename = os.path.basename(image_path).replace(".jpg", "_corrected.jpg")
#     save_path = os.path.join(output_folder, filename)
#     os.makedirs(output_folder, exist_ok=True)
#     cv2.imwrite(save_path, corrected_img)

#     R = corrected[2]  # OpenCV: BGR → R at index 2
#     creatinine_value = (208 - R) / 11.6
#     print(f"[RGB] Tip Foreground: R={R}, G={corrected[1]}, B={corrected[0]}")
#     print(f"[Save] Cropped & Corrected Image saved at: {save_path}")
#     return creatinine_value
