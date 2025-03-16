from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0005_userprofile_analytics_consent'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userprofile',
            name='highest_score',
        ),
    ]
