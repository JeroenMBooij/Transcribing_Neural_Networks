from src.apps.computer_vision.services.text_reader.data import preproc as pp
from src.apps.computer_vision.services.text_reader.data.compiled_model import CompiledModel

class TextReadService:

    @staticmethod
    def predict(text_image_file):

        input_size = (1024, 128, 1)
        img = pp.preprocess(text_image_file, input_size=input_size)
        x_test = pp.normalization([img])

        model = CompiledModel.get_instance(input_size)
        predicts, probabilities = model.predict(x_test, ctc_decode=True)

        predicts = [[model.tokenizer.decode(x) for x in y] for y in predicts][0]
        probabilities = probabilities[0].tolist()

        result = {}
        for i in range(len(predicts)):
            #we take the highest probability
            if(predicts[i] in result):
                value = result[predicts[i]]
                if(probabilities[i] > value):
                    result[predicts[i]] = probabilities[i] 
            else:
                result[predicts[i]] = probabilities[i]

        return result


