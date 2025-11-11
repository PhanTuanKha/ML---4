# do_an/churn_predict/core/urls.py

from django.urls import path
from . import views

app_name = 'core' # Giữ nguyên

# CHỈ CÓ MỘT DANH SÁCH DUY NHẤT
urlpatterns = [
    # Đường dẫn trang chủ
    path('', views.index_view, name='index_view'), 
    
    # Các đường dẫn trang ML
    path('predict/', views.predict_view, name='predict'),
    path('analysis/', views.analysis_view, name='analysis'),
    path('explanation/', views.explanation_view, name='explanation'),
    path('welcome/', views.welcome_view, name='welcome'),

    # Các đường dẫn API Auth
    path('api/register/', views.register_view, name='api_register'),
    path('api/login/', views.login_view, name='api_login'),
    path('logout/', views.logout_view, name='logout'),

    # Đường dẫn API Dataset
    path('api/get-dataset/', views.get_dataset_view, name='api_get_dataset'),

    # Đường dẫn API Quên mật khẩu
    path('api/forgot-verify/', views.forgot_password_verify_view, name='api_forgot_verify'),
    path('api/set-password/', views.set_new_password_view, name='api_set_password'),
    # Thêm 2 đường dẫn cho LIME và Upload
    path('api/explain-lime/', views.explain_lime_view, name='api_explain_lime'),
    path('upload-retrain/', views.upload_retrain_view, name='upload_retrain'),
]