import os
import cv2 #OpenCV for image processing-- read,write transform
import numpy as np# image handling
from pathlib import Path
from tensorflow.keras.datasets import mnist #Trial using prebuilt database

# Base folder containing 1.jpg … 6.jpg
BASE_DIR = Path(__file__).parent.absolute()
OUT_DIR  = os.path.join(BASE_DIR, "preprocessed_task1")
os.makedirs(OUT_DIR, exist_ok=True)

#preprocessing function area
def preprocess_single(img_bgr, out_size=128, pad=10):# expect 3channel BGR image- out size as 128x 128 pixel pad is just padding
    """
      Preprocess one image:
      1. Convert to grayscale
      2. Threshold to binary (digits white on black)
      3. Crop -> center pad -> resize to out_size (default 128x128)
      Returns: preprocessed image (uint8)
      """
    # --- Grayscale ---
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)#convert 3 channel BGR to 1 channel grayscale IN DOCUMENTATION REPORT LOGIC IS MOST DIGIT REC MODESL DONT NEED COLOUR
    # --- Adaptive threshold (digits white on black) ---
    bw = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               25, 10)# works like this: Convert to gray-> use a adaptive threshold to handle any uneven lighting(The simple thres was a asshole to work with) the thresh binary part inverts images so digits become WHITE(THIS IS A COMMON PRACTISE IN MACHINE LEARNING LOOK AT PAPER E4)
    #please remeber that this was done because white on black is easier for the model to read as featurtes stand otu more, also the 25 is blocksize /pxiel neighborhood and 10 is just constant subbed form the mean aka the fine tuna
    '''Remeber BW= Binary image ys=row indices of white pixels and xs= column indices of white pixels'''

    # --- Tight crop around digits ---
    ys, xs = np.where(bw > 0)# create  boolean array where TRUE when pixel is white //>0 and false while black  VITAL THIS FINDS ALL THE PXIELS THAT BELONG TO THE DIGIT AKA WE KNOW WHERE YOU SLEEP
    if xs.size and ys.size:#error prevent by checking if there are any white pixels at ALLS
        x0,x1=xs.min(),xs.max()#left edge of digit, right edge of digit
        y0,y1=ys.min(),ys.max()#top edge of digit, bottom edge of digit
        tight=bw[y0:y1+1,x0:x1+1]#Removes all unnessary black background and focuses only on the digit by cropping the image to the defined rectangle
    else:
        tight=bw # this is just if no white pixels found keep the og its just to avoid crashign and ensure array always returns

    # --- Center pad to square ---
    digit_height,digit_width=tight.shape # get the current height and width of the cropped digit Remeber teressact experiment
    canvas_side = max(digit_height, digit_width) + 2 * pad#Detrmine canvas size
    padded_canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)#make the empty
    # offset calculations to cetner the giit on the canvas
    yoffset=(canvas_side-digit_height)//2
    xoffset=(canvas_side -digit_width)//2
    padded_canvas[yoffset:yoffset+digit_height,xoffset:xoffset+digit_width]=tight#palce digit in center of canvas using offset cals from before
    # --- Resize to fixed size (default 128x128) ---
    resized = cv2.resize(padded_canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)# resize to 128x128 and the inter-area acts like scale feature in css,
    return resized
#ML models need consistent image sizing that is why it is resized that is important mention report in basic part.

#Load mnist  images and labels into training data
#This will allow near 60k trainign images and around 10k test images, without having to use the kraggle repository and enaring with crosiant
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("Training Images: ", x_train.shape)#x train is image of a digit
print("Training Labels: ",y_train.shape) # y_train is the label of the digit aka 0-9



#Loop over extensions, Build path, check file exists, return if path found else return none

def find_image(idx):
    """Return path for numbered file with common extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(BASE_DIR, f"{idx}{ext}")
      ##      return p
    return None

# ---------- Run preprocessing and save----------
which_nums_processing=100#SET TO 100 because its small  this is a test mech
for i in range(which_nums_processing):
    img_gray = x_train[i]#must be SQUARE LINES NOT THE BRACKETS OR ELSE IT WILL SAY OBECJT NOT CALLABLE
    #convert to BGR so its compatible with the preprocesser function
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    #send to preprocess
    pre_img = preprocess_single(img_bgr, out_size=128)   # 👈 made images bigger

    # Save output PNG
    out_png = os.path.join(OUT_DIR, f"mnist{i}_preprocessed.png")
    cv2.imwrite(out_png, pre_img)

    print(f"[OK] Image {i}: saved {out_png}")

print(f"\nTask 1 complete! Preprocessed images are in: {OUT_DIR}")


