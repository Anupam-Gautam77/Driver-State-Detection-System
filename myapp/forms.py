# forms.py

from django import forms
from .models import Activity
from .models import Video

class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video_file','vtitle']
        widgets = {
            'vtitle': forms.TextInput(attrs={'class': 'form-control mb-2', 'placeholder': 'Enter title', 'style': 'border: 2px solid #ccc; padding: 10px; border-radius: 5px;'}),
            'video_file': forms.FileInput(attrs={'class': 'form-control-file mt-2', 'style': 'border: 2px solid #ccc; padding: 10px; border-radius: 5px;'}),
        }
        

class ActivityForm(forms.ModelForm):
    class Meta:
        model = Activity
        fields = ['title', 'file']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control mb-2', 'placeholder': 'Enter title', 'style': 'border: 2px solid #ccc; padding: 10px; border-radius: 5px;'}),
            'file': forms.FileInput(attrs={'class': 'form-control-file mt-2', 'style': 'border: 2px solid #ccc; padding: 10px; border-radius: 5px;'}),
        }