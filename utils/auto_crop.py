# import cv2
# import numpy as np
# import os
# import uuid
# from config.settings import RESIZED_WIDTH, RESIZED_HEIGHT

# def auto_crop_and_resize(image_path):
#     image = cv2.imread(image_path)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     mask1 = cv2.inRange(hsv, (0, 50, 50), (20, 255, 255))
#     mask2 = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
#     mask = cv2.bitwise_or(mask1, mask2)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
#     mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise Exception("No liquid region detected.")

#     largest = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest)

#     cropped = image[y:y+h, x:x+w]
#     resized = cv2.resize(cropped, (RESIZED_WIDTH, RESIZED_HEIGHT))

#     output_path = os.path.join("inputs", f"{uuid.uuid4().hex[:8]}_cropped.jpg")
#     cv2.imwrite(output_path, resized)
#     return output_path




import cv2
import numpy as np
from config.settings import RESIZED_WIDTH, RESIZED_HEIGHT  # Import target dimensions from settings

def detect_liquid_region(img):
    """
    Detects yellowish or reddish liquid solution regions in the image using HSV thresholds.
    Returns a binary mask indicating potential liquid regions.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert BGR image to HSV color space

    # Define HSV ranges for yellow and red shades
    yellow_lower = np.array([20, 40, 40])
    yellow_upper = np.array([40, 255, 255])

    red_lower1 = np.array([0, 40, 40])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 40, 40])
    red_upper2 = np.array([180, 255, 255])

    # Create binary masks for yellow and red regions
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)

    # Combine red and yellow masks into a single mask
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    combined_mask = cv2.bitwise_or(yellow_mask, red_mask)

    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    return combined_mask

def auto_crop(image_path):
    """
    Automatically detects the liquid region in the image, crops it around the detected region,
    resizes the cropped image to (RESIZED_WIDTH x RESIZED_HEIGHT), and saves the output.
    Returns the path to the cropped image.
    """
    img = cv2.imread(image_path)  # Load the image
    if img is None:
        raise ValueError("Could not read image")

    # Apply blur to reduce noise and improve mask quality
    img = cv2.GaussianBlur(img, (15, 15), 0)

    # Get the mask for red/yellow liquid regions
    mask = detect_liquid_region(img)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour as the region of interest
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ignore very small detections to avoid false positives
        if w * h < 500:
            print("[AutoCrop] Region too small — skipping contour crop")
            return image_path

        # Adjust the y-axis crop to focus on the lower portion of the detected region
        y += int(0.25 * h)
        h = int(0.35 * h)

        # Expand width for better context on both sides
        extra_w = int(0.5 * w)
        x = max(0, x - extra_w)
        w = min(img.shape[1] - x, w + 2 * extra_w)

        # Crop and resize the region
        cropped_img = img[y:y + h, x:x + w]
        resized = cv2.resize(cropped_img, (RESIZED_WIDTH, RESIZED_HEIGHT), interpolation=cv2.INTER_AREA)

        # Save the cropped and resized image
        temp_path = image_path.replace(".jpg", "_cropped.jpg").replace(".png", "_cropped.png")
        cv2.imwrite(temp_path, resized)
        return temp_path

    # If no contours detected, return the original image
    print("[AutoCrop] No region detected — returning original image")
    return image_path
