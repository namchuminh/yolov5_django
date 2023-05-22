from django.urls import path
from .views import video_feed, index

urlpatterns = [
    path('video_feed/', video_feed, name='video_feed'),
    path('', index, name='index')
]
