# Generated by Django 4.1.5 on 2024-02-21 09:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0003_activity_title'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='activity',
            name='title',
        ),
    ]
