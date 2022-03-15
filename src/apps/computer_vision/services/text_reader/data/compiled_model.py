import string
from src.apps.computer_vision.services.text_reader.data.generator import Tokenizer
from src.apps.computer_vision.services.text_reader.network.model import HTRModel


class CompiledModel:
    instance = None
    model = None
    tokenizer = None


    def __init__(self, input_size):
        if CompiledModel.instance is None:

            self.updateModel(input_size)

            CompiledModel.instance = self

    
    @staticmethod 
    def get_instance(tokenizer, input_size):
      if CompiledModel.instance is None:
          CompiledModel(tokenizer, input_size)
      
      return CompiledModel.instance

    def predict(self, x_test, ctc_decode):
        return self.model.predict(x_test, ctc_decode=ctc_decode)

    def updateModel(self, input_size):
        if tokenizer is None:
            tokenizer = Tokenizer(chars=string.printable[:95], max_text_length=128)
        
        self.model = HTRModel(architecture='flor',
                            input_size=input_size,
                            vocab_size=tokenizer.vocab_size,
                            beam_width=10,
                            top_paths=10)

        self.model.compile(learning_rate=0.001)
        self.model.load_checkpoint(target='src/apps/computer_vision/services/text_reader/files/handwritten_text_model.hdf5')
        
