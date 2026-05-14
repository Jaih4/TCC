from django.shortcuts import render
from api.models import Comment

def dashboard(request):
    comments = Comment.objects.all()
    return render(request, 'dashboard/index.html', {'comments': comments})