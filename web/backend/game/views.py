from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.db import transaction
from django.db.models import Q
from .game_manager import GameManager
from .models import GameRecord, UserProfile, Match
from .serializers import (
    UserSerializer, 
    UserProfileSerializer, 
    GameRecordSerializer,
    UserRegistrationSerializer,
    MatchSerializer,
    CreateMatchSerializer
)
from uuid import uuid4


class GameViewSet(viewsets.ViewSet):
    @action(detail=False, methods=["post"])
    def create_game(self, request):
        game_id = str(uuid4())
        agent_type = request.data.get("agent_type", "rule")
        player_color = request.data.get("player_color", "black")
        user = request.user if request.user.is_authenticated else None
        
        observation, actual_player_color = GameManager.create_game(game_id, agent_type, user, player_color)
        request.session["game_id"] = game_id
        return Response({
            "observation": observation, 
            "game_id": game_id,
            "player_color": actual_player_color
        })

    @action(detail=False, methods=["post"])
    def make_move(self, request):
        game_id = request.data.get("game_id")
        move = request.data.get("move")
        
        print(f"make_move called with game_id={game_id}, move={move}")
        
        try:
            match = Match.objects.get(game__game_id=game_id)
            is_multiplayer = True
            print(f"Found multiplayer match: {match}")
        except:
            is_multiplayer = False
            print(f"No multiplayer match found with game_id={game_id}")
        
        print(f"Available single-player games: {list(GameManager._games.keys())}")
        
        if is_multiplayer:
            print(f"Handling as multiplayer game. User: {request.user.username if request.user.is_authenticated else 'Anonymous'}")
            if not request.user.is_authenticated:
                return Response({"error": "Authentication required for multiplayer"}, status=status.HTTP_401_UNAUTHORIZED)
            
            try:
                print(f"Found match: {match}, current_turn: {match.current_turn.username if match.current_turn else 'None'}")
                print(f"Is user's turn: {match.current_turn == request.user}")
                
                result = GameManager.make_multiplayer_move(game_id, move, request.user)
                return Response(result)
            except ValueError as e:
                print(f"ValueError in make_multiplayer_move: {str(e)}")
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                import traceback
                print(f"Exception in make_multiplayer_move: {str(e)}")
                print(traceback.format_exc())
                return Response({"error": f"Server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            print(f"Handling as single-player game")
            print(f"GameManager._games: {GameManager._games}")
            if not game_id or game_id not in GameManager._games:
                return Response({"error": "No active game"}, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                result = GameManager.make_move(game_id, move)
                return Response(result)
            except Exception as e:
                print(f"Exception in make_move: {str(e)}")
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class UserViewSet(viewsets.ViewSet):
    @action(detail=False, methods=["post"])
    def register(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        
        if serializer.is_valid():
            with transaction.atomic():
                user = serializer.save()
                token, _ = Token.objects.get_or_create(user=user)
                
            return Response({
                "token": token.key,
                "user": UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=["post"])
    def login(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        
        user = authenticate(username=username, password=password)
        
        if user:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                "token": token.key,
                "user": UserSerializer(user).data
            })
            
        return Response(
            {"error": "Invalid credentials"}, 
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    @action(detail=False, methods=["get"])
    def profile(self, request):
        if not request.user.is_authenticated:
            return Response(
                {"error": "Authentication required"}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
            
        try:
            profile = request.user.profile
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data)
        except UserProfile.DoesNotExist:
            profile = UserProfile.objects.create(user=request.user)
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data)
    
    @action(detail=False, methods=["get"])
    def game_history(self, request):
        if not request.user.is_authenticated:
            return Response(
                {"error": "Authentication required"}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
            
        games = GameRecord.objects.filter(user=request.user).order_by('-start_time')
        serializer = GameRecordSerializer(games, many=True)
        return Response(serializer.data)


class MatchViewSet(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=False, methods=["post"])
    def create_match(self, request):
        serializer = CreateMatchSerializer(data=request.data, context={'request': request})
        
        if serializer.is_valid():
            game_id = str(uuid4())
            game_record = GameRecord.objects.create(
                user=request.user,
                game_id=game_id,
                opponent_type='multiplayer'
            )
            print(f"Created game record with ID: {game_record.id}, game_id: {game_id}")
            
            match = serializer.save(
                creator=request.user,
                game=game_record,
                status="open"
            )
            print(f"Match created in DB: {match}")
            
            match_data = GameManager.create_multiplayer_match(match)
            print(f"Match initialized in game manager: {match_data}")
            
            return Response({
                "match": MatchSerializer(match, context={'request': request}).data,
                "game_state": match_data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=["get"])
    def list_open_matches(self, request):
        query = Match.objects.filter(status="open").exclude(creator=request.user)
            
        matches = query.order_by('-created_at')
        
        serializer = MatchSerializer(matches, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=["get"])
    def my_matches(self, request):
        print(f"Getting matches for user: {request.user.username}")
        print(f"Raw matches: {Match.objects.all()}")
        
        matches = Match.objects.filter(
            status__in=["open", "matched", "in_progress", "completed"]
        ).filter(
            creator=request.user
        ) | Match.objects.filter(
            status__in=["matched", "in_progress", "completed"]
        ).filter(
            opponent=request.user
        ).order_by('-updated_at')
        
        serializer = MatchSerializer(matches, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=["post"])
    def join_match(self, request, pk=None):
        print(f"join_match called for match_id={pk} by user={request.user.username}")
        
        try:
            match = Match.objects.get(pk=pk, status="open")
            print(f"Found match to join: {match}")
        except Match.DoesNotExist:
            return Response(
                {"error": "Match not found or not available"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if match.creator == request.user:
            return Response(
                {"error": "Cannot join your own match"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            match_data = GameManager.join_match(match, request.user)
            print(f"Join successful. Match data: {match_data}")
            
            return Response({
                "match": MatchSerializer(match, context={'request': request}).data,
                "game_state": match_data
            })
        except ValueError as e:
            print(f"Error joining match: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            import traceback
            print(f"Exception in join_match: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": f"Server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=["get"])
    def match_state(self, request, pk=None):
        try:
            
            match = Match.objects.filter(
                pk=pk
            ).filter(
                Q(creator=request.user) | Q(opponent=request.user)
            ).first()
            
            if not match:
                return Response(
                    {"error": "Match not found or you are not a participant"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            match_data = GameManager.get_match_state(match, request.user)
            
            return Response({
                "match": MatchSerializer(match, context={'request': request}).data,
                "game_state": match_data
            })
        except Exception as e:
            import traceback
            print(f"Error in match_state: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=["post"])
    def forfeit_match(self, request, pk=None):
        try:
            from django.db.models import Q
            
            match = Match.objects.filter(
                pk=pk, 
                status="in_progress"
            ).filter(
                Q(creator=request.user) | Q(opponent=request.user)
            ).first()
            
            if not match:
                return Response(
                    {"error": "Match not found, not in progress, or you are not a participant"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            match.status = "completed"
            match.game.result = "loss"
            match.game.final_score = 0
            match.game.save()
            black_player = match.black_player
            white_player = match.white_player
            
            if black_player and white_player:
                is_black = black_player == request.user
                winner = 1 if is_black else 0 
                
                try:
                    black_profile = black_player.profile
                    white_profile = white_player.profile
                    
                    black_profile.games_played += 1
                    white_profile.games_played += 1
                    
                    if winner == 0: 
                        black_profile.wins += 1
                        white_profile.losses += 1
                        print(f"Black ({black_player.username}) won by forfeit against White ({white_player.username})")
                    else:  
                        white_profile.wins += 1
                        black_profile.losses += 1
                        print(f"White ({white_player.username}) won by forfeit against Black ({black_player.username})")
                    
                    black_profile.save()
                    white_profile.save()
                except Exception as profile_error:
                    print(f"Error updating profiles: {str(profile_error)}")
            
            match.save()
            
            return Response({"status": "Match forfeited", "match": MatchSerializer(match, context={'request': request}).data})
        except Exception as e:
            import traceback
            print(f"Error in forfeit_match: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
