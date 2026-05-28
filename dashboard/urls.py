from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('database/', views.database_dashboard, name='database_dashboard'),
    path('dashboard-nlp/',views.dashboard_nlp,name='dashboard_nlp'),
    path('post/', views.analisar_post_especifico_view, name='post_especifico'),
]