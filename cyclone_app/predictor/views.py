from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import UploadImageForm
from .predict import predict_intensity

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            result = predict_intensity(image)
            return render(request, 'predictor/result.html', {'result': result})
    else:
        form = UploadImageForm()
    return render(request, 'predictor/upload.html', {'form': form})
