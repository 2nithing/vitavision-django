from django.shortcuts import render
from .forms import ImageUploadForm
from .utils import predict_vitamin

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        # Save the file temporarily
        image_path = f'media/{uploaded_file.name}'
        with open(image_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        # Predict the vitamin deficiency
        prediction = predict_vitamin(image_path)
        
        # Return the result to the template
        return render(request, 'home.html', {'form': ImageUploadForm(), 'prediction': prediction})
    
    return render(request, 'home.html', {'form': ImageUploadForm()})
