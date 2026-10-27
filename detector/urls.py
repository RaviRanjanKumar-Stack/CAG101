#creating urls for webpages
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_csv, name="upload_csv"),
]
