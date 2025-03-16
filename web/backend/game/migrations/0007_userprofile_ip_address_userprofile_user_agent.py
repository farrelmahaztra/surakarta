from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0006_remove_userprofile_highest_score'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='ip_address',
            field=models.GenericIPAddressField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='userprofile',
            name='user_agent',
            field=models.TextField(blank=True, null=True),
        ),
    ]
