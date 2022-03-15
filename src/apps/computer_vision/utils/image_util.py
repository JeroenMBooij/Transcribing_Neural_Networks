from PIL import ImageFile
from keras.preprocessing import image
import cv2 as cv

class ImageUtil:
    
    @staticmethod
    def request_image_to_PIL(image_file):

        imageParser = ImageFile.Parser()

        while True:
            s = image_file.read(1024)
            if not s:
                break
            imageParser.feed(s)
        
        PIL_image = imageParser.close()

        return PIL_image

    @staticmethod
    def image_to_gray_scale_format(PIL_image):
        
        PIL_image = PIL_image.resize((28, 28))
        image_array = image.img_to_array(PIL_image)

         # if the image is already gray scale COLOR_BGR2GRAY won't work so we can just use the original image
        try:
            gray_image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
        except:
          gray_image_array = image_array

        gray_image_array = gray_image_array.reshape(1, 28, 28, 1)

        return gray_image_array
