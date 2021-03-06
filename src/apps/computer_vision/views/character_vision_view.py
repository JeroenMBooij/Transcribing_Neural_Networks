import os
from django.http.response import HttpResponse
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema
from django.http import HttpResponse

from src.apps.computer_vision.services.character_reader.character_read_service import CharacterReadService
from src.apps.computer_vision.utils.image_util import ImageUtil


class AllCharactersVisionView(APIView):

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
    )
    def post(self, request):

        source_path = os.path.join('src', 'apps', 'computer_vision', 'services','character_reader', 'files', 'models')
        if os.path.exists(f'{source_path}/all_handwritten_characters_model.hdf5') == False:
            return HttpResponse('No character NN Checkpoint found', status=400)


        character_image = ImageUtil.request_image_to_PIL(request.FILES['file'])

        prediction = CharacterReadService.predict_all(character_image)

        return HttpResponse(prediction, status=201)
    
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
        description='Add new Keras checkpoint (hdf5) for the computer vision character neural network',
    )
    def put(self, request):

        output_path = os.path.join('src', 'apps', 'computer_vision', 'services','character_reader', 'files', 'models')
        data_file = request.FILES['file']
        
        file_name, file_extension = os.path.splitext(data_file.name)
        if file_extension != ".hdf5":
            return HttpResponse(status=400)

        with open(f'{output_path}/all_handwritten_characters_model.hdf5', 'wb+') as destination:
            for chunk in data_file.chunks():
                destination.write(chunk)

        return HttpResponse(status=201)