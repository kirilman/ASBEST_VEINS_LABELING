
import os
import numpy as np
import json
from PIL import Image


#-----------------------------------

def _imgs2gray(img_pths):
    '''Convert Image to gray scale format.
    Parameters
    -----------
    img_pths: list[string],
      list of image pathes 2 check and resize.
    
    Returns
    --------
    list[string],
      list of corrected images
    '''    
    report_list = list()
    for img_pth in img_pths:
        img = Image.open(img_pth)
        if Image.open(img_pth).mode != 'L':
            img.convert("L").save(img_pth)
            report_list.append(img_pth)
    return report_list
#-----------------------------------

#----------------------------------

def _resize_imgs(img_pths, width, height):
    ''' Resize Image by list of pathes
    Parameters
    -----------
    img_pths: list[string],
      list of image pathes 2 check and resize.
    width, height: int, int,
      image width and height
    
    Returns
    --------
    list[string],
      list of corrected images
    '''
    report_list = list()
    for img_pth in img_pths:
        img = Image.open(img_pth)
        width_, height_ = img.size
        if width_ != width or height_ != height:
            img = img.resize((width, height), Image.ANTIALIAS)
            img.save(img_pth)
            report_list.append(img_pth)
    return report_list

#----------------------------------
def convert2jpeg(folder):
  for filename in os.listdir(folder):
      if filename.lower().endswith(".png"):
          png_path = os.path.join(folder, filename)
          jpg_path = os.path.splitext(png_path)[0] + ".jpg"
          with Image.open(png_path) as img:
              # Конвертируем в RGB, заменяя прозрачность на белый
              if img.mode in ("RGBA", "LA", "P"):
                  background = Image.new("RGB", img.size, (255, 255, 255))
                  background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                  img = background
              elif img.mode != "RGB":
                  img = img.convert("RGB")
              img.save(jpg_path, "JPEG", quality=100)


