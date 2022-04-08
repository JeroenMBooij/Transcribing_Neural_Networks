import os
import sys
from django.http.response import HttpResponse
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema
from django.http import HttpResponse, JsonResponse
from src.apps.computer_vision.services.text_reader.text_read_service import TextReadService

from src.apps.computer_vision.utils.image_util import ImageUtil


class TextVisionView(APIView):

    @extend_schema(
        description='Upload an image with a height of 90 pixels of (a) handwritten word(s)',
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
        responses = { 
            201: {
                "results": {
                    "predictions": [],
                    "probabilties": []
                }
            }
        }
    )
    def post(self, request):

        source_path = os.path.join('src', 'apps', 'computer_vision', 'services','text_reader', 'files')
        if os.path.exists(f'{source_path}/handwritten_text_model.hdf5') == False:
            return HttpResponse('No character NN Checkpoint found', status=400)

        print(request.FILES, file=sys.stderr)

        prediction = TextReadService.predict(request.FILES['file'])

        return JsonResponse(prediction, status=201)


    
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
        description='Add new Keras checkpoint (hdf5) for the computer vision neural network',
    )
    def put(self, request):

        output_path = os.path.join('src', 'apps', 'computer_vision', 'services','text_reader', 'files')
        data_file = request.FILES['file']
        
        file_name, file_extension = os.path.splitext(data_file.name)
        if file_extension != ".hdf5":
            return HttpResponse(status=400)

        with open(f'{output_path}/handwritten_text_model.hdf5', 'wb+') as destination:
            for chunk in data_file.chunks():
                destination.write(chunk)

        return HttpResponse(status=201)