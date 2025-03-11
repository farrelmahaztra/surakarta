from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
from django.utils import timezone

class GameRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games')
    game_id = models.CharField(max_length=36, unique=True)
    opponent_type = models.CharField(max_length=20, default='rule')
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    
    moves = JSONField(default=list)
    final_score = models.IntegerField(null=True, blank=True)
    result = models.CharField(max_length=10, null=True, blank=True)  
    
    def __str__(self):
        return f"Game {self.game_id} - {self.user.username}"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    games_played = models.IntegerField(default=0)
    wins = models.IntegerField(default=0)
    losses = models.IntegerField(default=0)
    draws = models.IntegerField(default=0)
    highest_score = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Profile for {self.user.username}"


class Match(models.Model):
    MATCH_STATUS_CHOICES = [
        ('open', 'Open'),
        ('matched', 'Matched'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
    ]
    
    PLAYER_COLOR_CHOICES = [
        ('black', 'Black'),
        ('white', 'White'),
        ('random', 'Random'),
    ]
    
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_matches')
    opponent = models.ForeignKey(User, on_delete=models.CASCADE, related_name='joined_matches', null=True, blank=True)
    game = models.ForeignKey(GameRecord, on_delete=models.CASCADE, related_name='matches', null=True, blank=True)
    
    creator_color = models.CharField(max_length=10, choices=PLAYER_COLOR_CHOICES, default='random')
    status = models.CharField(max_length=15, choices=MATCH_STATUS_CHOICES, default='open')
    
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    current_turn = models.ForeignKey(User, on_delete=models.CASCADE, related_name='turns', null=True, blank=True)
    last_activity = models.DateTimeField(auto_now=True)
    
    board_state = JSONField(default=dict)
    moves_history = JSONField(default=list)
    
    black_player = models.ForeignKey(User, on_delete=models.CASCADE, related_name='black_matches', null=True, blank=True)
    white_player = models.ForeignKey(User, on_delete=models.CASCADE, related_name='white_matches', null=True, blank=True)
    
    current_player_idx = models.IntegerField(default=0)
    
    def __str__(self):
        opponent_name = self.opponent.username if self.opponent else "Waiting for opponent"
        return f"Match {self.game_id}: {self.creator.username} vs {opponent_name} ({self.status})"
