import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Match',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('game_id', models.CharField(max_length=36, unique=True)),
                ('creator_color', models.CharField(choices=[('black', 'Black'), ('white', 'White'), ('random', 'Random')], default='random', max_length=10)),
                ('status', models.CharField(choices=[('open', 'Open'), ('matched', 'Matched'), ('in_progress', 'In Progress'), ('completed', 'Completed'), ('abandoned', 'Abandoned')], default='open', max_length=15)),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('last_activity', models.DateTimeField(auto_now=True)),
                ('board_state', models.JSONField(default=dict)),
                ('moves_history', models.JSONField(default=list)),
                ('creator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='created_matches', to=settings.AUTH_USER_MODEL)),
                ('current_turn', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='turns', to=settings.AUTH_USER_MODEL)),
                ('opponent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='joined_matches', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
