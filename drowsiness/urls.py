from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static
from django.conf import settings
from. import views
app_name = 'drowsiness'
urlpatterns = [
   
   
      # path('d_dtcn/', views.detection, name='detection'),
      path('drowsy/', views.drowsy, name='drowsy'),
    
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
