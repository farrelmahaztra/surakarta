from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import GameViewSet, UserViewSet, MatchViewSet

router = DefaultRouter()
router.register(r"game", GameViewSet, basename="game")
router.register(r"users", UserViewSet, basename="users")
router.register(r"matches", MatchViewSet, basename="matches")

urlpatterns = [
    path("", include(router.urls)),
]
