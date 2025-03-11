from rest_framework import serializers
from django.contrib.auth.models import User
from .models import GameRecord, UserProfile, Match


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email')
        extra_kwargs = {'password': {'write_only': True}}


class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ('username', 'games_played', 'wins', 'losses', 'draws', 'highest_score')


class GameRecordSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    opponent_name = serializers.SerializerMethodField(read_only=True)
    
    class Meta:
        model = GameRecord
        fields = ('id', 'game_id', 'username', 'opponent_type', 'opponent_name', 'start_time', 
                  'end_time', 'final_score', 'result')
                  
    def get_opponent_name(self, obj):
        if obj.opponent_type == 'multiplayer':
            try:
                match = obj.matches.first() 
                if match:
                    if match.creator == obj.user:
                        return match.opponent.username if match.opponent else "Unknown"
                    else:
                        return match.creator.username
            except Exception:
                pass
        
        return None


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password')
    
    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password']
        )
        UserProfile.objects.create(user=user)
        return user


class MatchSerializer(serializers.ModelSerializer):
    creator_username = serializers.CharField(source='creator.username', read_only=True)
    opponent_username = serializers.CharField(source='opponent.username', read_only=True)
    current_turn_username = serializers.CharField(source='current_turn.username', read_only=True)
    game_record_id = serializers.SerializerMethodField(read_only=True)
    
    def get_game_record_id(self, obj):
        return obj.game.id if obj.game else None
    
    class Meta:
        model = Match
        fields = ('id', 'game_id', 'creator_username', 'opponent_username', 
                  'creator_color', 'status', 'created_at', 'updated_at',
                  'current_turn_username', 'last_activity', 'game_record_id')

class CreateMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = Match
        fields = ('creator_color',)
        
    def create(self, validated_data):
        validated_data['creator'] = self.context['request'].user
        return super().create(validated_data)