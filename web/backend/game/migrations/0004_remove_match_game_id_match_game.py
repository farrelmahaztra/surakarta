import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0003_match_black_player_match_current_player_idx_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='match',
            name='game_id',
        ),
        migrations.AddField(
            model_name='match',
            name='game',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='matches', to='game.gamerecord'),
        ),
    ]
