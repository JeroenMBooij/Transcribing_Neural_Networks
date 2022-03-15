import os
from django.http.response import HttpResponse
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema
from django.http import HttpResponse
from src.apps.computer_vision.services.text_reader.text_train_service import TextTrainService
import threading



class TextTrainingView(APIView):

    @extend_schema(
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'file': {
                        'type': 'string',
                        'format': 'binary'
                    }
                }
            }
        },
        description='Add new data (hdf5) and retrain the computer vision neural network',
    )
    def patch(self, request):
        
        trainerthreads = [t for t in threading.enumerate() if t.name == "Trainer" and t.is_alive()]
        if(trainerthreads):
            return HttpResponse(status=503)

        output_path = os.path.join('src', 'apps', 'computer_vision', 'services','text_reader', 'files')
        data_file = request.FILES['file']
        
        with open(f'{output_path}/{data_file.name}', 'wb+') as destination:
            for chunk in data_file.chunks():
                destination.write(chunk)

        train_thread = threading.Thread(target=TextTrainService.start, name="Trainer", args=[output_path, data_file.name])
        train_thread.start()

        return HttpResponse(status=200)