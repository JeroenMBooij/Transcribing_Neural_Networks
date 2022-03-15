from django.http.response import HttpResponse
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema
from django.http import HttpResponse, JsonResponse
from src.apps.computer_vision.services.text_reader.text_read_service import TextReadService

from src.apps.computer_vision.utils.image_util import ImageUtil


class TextVisionView(APIView):

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

        prediction = TextReadService.predict(request.FILES['file'])

        return JsonResponse({"results": prediction}, status=201)