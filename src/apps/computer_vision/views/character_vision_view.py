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

        character_image = ImageUtil.request_image_to_PIL(request.FILES['file'])

        prediction = CharacterReadService.predict_all(character_image)

        return HttpResponse(prediction, status=201)