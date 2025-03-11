from django.contrib import admin
from .models import GameRecord, UserProfile

@admin.register(GameRecord)
class GameRecordAdmin(admin.ModelAdmin):
    list_display = ('game_id', 'user', 'opponent_type', 'start_time', 'end_time', 'result')
    list_filter = ('opponent_type', 'result')
    search_fields = ('game_id', 'user__username')

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'games_played', 'wins', 'losses', 'draws', 'highest_score')
    search_fields = ('user__username',)
