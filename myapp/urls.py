from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static
from django.conf import settings
app_name = 'myapp'
urlpatterns = [
   
     path('',views.index,name='index'),
#      path('request', views.index1, name='request'),
    # path('classify/', classify_image, name='classify_image'),
    # path('upload', upload_image, name='upload'),
    path('update_activity', views.update_activity, name='update_activity'),
    path('update_video', views.update_video, name='update_video'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
