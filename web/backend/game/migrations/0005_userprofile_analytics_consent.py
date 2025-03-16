from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0004_remove_match_game_id_match_game'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='analytics_consent',
            field=models.BooleanField(default=False),
        ),
    ]
