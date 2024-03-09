# models.py

from django.db import models

class Activity(models.Model):
    title = models.TextField(default="Images")
    file = models.ImageField(upload_to='media/', default='default_image.png')  # Replace 'default_image.png' with the path to your default image
class Video(models.Model):
    video_file = models.FileField(upload_to='media/')   
    vtitle = models.TextField(default="videos")
def __str__(self):
        return self.file.name