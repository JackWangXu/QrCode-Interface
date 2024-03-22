from django.urls import path
from QrCodeServer.views import generate_image

urlpatterns = [
    path('generate_image', generate_image, name='generate_image'),
]
