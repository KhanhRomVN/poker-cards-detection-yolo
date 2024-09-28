from django.urls import path
from . import views

urlpatterns = [
    path('poker-card-detection/', views.poker_card_detection, name='poker_card_detection'),
]