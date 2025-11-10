from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('analysis/', views.analysis_view, name='analysis'),
    path('explanation/', views.explanation_view, name='explanation'),
    path('welcome/', views.welcome_view, name='welcome'),
    # Thêm các URL cho login, signup, forgot-password ở đây
]