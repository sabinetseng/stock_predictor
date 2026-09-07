from django.urls import path
from . import views

app_name = "market_data"

urlpatterns = [
    path("index/<str:index_code>/", views.index_history, name="index_history"),
]
