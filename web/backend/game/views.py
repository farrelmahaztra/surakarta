from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .game_manager import GameManager
from uuid import uuid4


class GameViewSet(viewsets.ViewSet):
    @action(detail=False, methods=["post"])
    def create_game(self, request):
        game_id = str(uuid4())
        agent_type = request.data.get("agent_type", "rule")
        observation = GameManager.create_game(game_id, agent_type)
        request.session["game_id"] = game_id
        return Response({"observation": observation, "game_id": game_id})

    @action(detail=False, methods=["post"])
    def make_move(self, request):
        game_id = request.data.get("game_id")
        if not game_id or game_id not in GameManager._games:
            return Response({"error": "No active game"}, status=400)

        move = request.data.get("move")
        result = GameManager.make_move(game_id, move)
        return Response(result)
