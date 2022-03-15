from django.urls import path
from .views import TextTrainingView
from .views import TextVisionView
from .views import AllCharactersVisionView

urlpatterns = [
    path('read/all-characters/', AllCharactersVisionView.as_view()),
    path('read/text/', TextVisionView.as_view()),
    path('train/text/', TextTrainingView.as_view())
]