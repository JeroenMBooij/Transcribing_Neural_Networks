import datetime
import string
import os
from src.apps.computer_vision.services.text_reader.data.compiled_model import CompiledModel
from src.apps.computer_vision.services.text_reader.data.generator import DataGenerator
from src.apps.computer_vision.services.text_reader.network.model import HTRModel
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


class TextTrainService:

    @staticmethod
    def start(output_path, datasource_name):

        input_size = (1024, 128, 1)
        arch = 'flor'

        #TODO implement didac source (instead of iam) in DataGenerator
        dtgen = DataGenerator(source=os.path.join(output_path, datasource_name),
                              batch_size=16,
                              charset=string.printable[:95],
                              max_text_length=128)

        model = HTRModel(architecture=arch,
                         input_size=input_size,
                         vocab_size=dtgen.tokenizer.vocab_size,
                         beam_width=10,
                         stop_tolerance=20,
                         reduce_tolerance=15)
        model.compile(learning_rate=0.001)

        model.summary(output_path, f"{arch}.summary.txt")
        callbacks = model.get_callbacks(logdir=output_path, checkpoint=output_path, verbose=1)

        start_time = datetime.datetime.now()
        h = model.fit(x=dtgen.next_train_batch(),
                        epochs=1000,
                        steps_per_epoch=dtgen.steps['train'],
                        validation_data=dtgen.next_valid_batch(),
                        validation_steps=dtgen.steps['valid'],
                        callbacks=callbacks,
                        shuffle=True,
                        verbose=1)
        total_time = datetime.datetime.now() - start_time

        TextTrainService._log_training_results(h, total_time, dtgen, output_path, arch)

        newModel = CompiledModel.get_instance(input_size)
        newModel.update(input_size)

    

    @staticmethod
    def _log_training_results(h, total_time, dtgen, output_path, arch):
        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))
        total_item = (dtgen.size['train'] + dtgen.size['valid'])

        t_corpus = "\n".join([
            f"Total train images:      {dtgen.size['train']}",
            f"Total validation images: {dtgen.size['valid']}",
            f"Batch:                   {dtgen.batch_size}\n",
            f"Total time:              {total_time}",
            f"Time per epoch:          {time_epoch}",
            f"Time per item:           {time_epoch / total_item}\n",
            f"Total epochs:            {len(loss)}",
            f"Best epoch               {min_val_loss_i + 1}\n",
            f"Training loss:           {loss[min_val_loss_i]:.8f}",
            f"Validation loss:         {min_val_loss:.8f}"
        ])

        with open(os.path.join(output_path, f"{arch}.train.txt"), "w") as lg:
            lg.write(t_corpus)


