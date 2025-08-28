from django.urls import path
from .views import optimize_portfolio

urlpatterns = [
    path("optimize/", optimize_portfolio),
]
