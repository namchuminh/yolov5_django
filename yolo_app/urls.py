from django.urls import path
from .views import video_feed, index, send_info

urlpatterns = [
    path('video_feed/', video_feed, name='video_feed'),
    path('send_info/', send_info, name='send_info'),
    path('', index, name='index'),
]
