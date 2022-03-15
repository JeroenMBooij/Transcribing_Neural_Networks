from keras.models import load_model
import pandas as pd
from src.apps.computer_vision.utils.image_util import ImageUtil

class CharacterReadService:

    @staticmethod
    def predict_all(character_image):

        gray_character_array = ImageUtil.image_to_gray_scale_format(character_image)

        # This model is trained in Google Colab (https://colab.research.google.com/drive/16-2RFIBQ-y21aDFI6yFODOpm-GVb44J_)
        model = load_model('src/apps/computer_vision/services/character_reader/files/models/all_handwritten_characters_model.hdf5')
        cl = model.predict(gray_character_array)
        cl = list(cl[0])

        acii_all_map = pd.read_csv("src/apps/computer_vision/services/character_reader/files/maps/ascii_all_map.csv")
        prediction = acii_all_map["Character"][cl.index(max(cl))]

        return prediction

