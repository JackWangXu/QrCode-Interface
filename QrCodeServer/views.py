import json
from django.http import HttpResponse, JsonResponse
from PIL import Image
import qrcode
from django.views.decorators.csrf import csrf_exempt
import io

from .image_generator import inference

@csrf_exempt
def generate_image(request):
    print(request)
    if request.method == "POST":
        try:
            print("POST data:", request.POST)
            print("FILES data:", request.FILES)
            qr_code_content = request.POST.get('qr_code_content', '')
            prompt = request.POST.get('prompt', '')
            negative_prompt = request.POST.get('negative_prompt', '')
            controlnet_conditioning_scale = request.POST.get('controlnet_conditioning_scale', '')
            guidance_scale = request.POST.get('guidance_scale', '')
            strength = request.POST.get('strength', '')
            seed = request.POST.get('seed', '')

            init_image = None
            if 'init_image' in request.FILES:
                print(888888888888888888888)
                init_image_file = request.FILES['init_image']
                init_image = Image.open(init_image_file)
                image = inference(
                    qr_code_content=qr_code_content,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    init_image=init_image,
                    controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                    guidance_scale=float(guidance_scale),
                    strength=float(strength),
                    seed=int(seed)
                )
            else:
                print(99999999999999999)
                image = inference(
                    qr_code_content=qr_code_content,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    init_image=init_image,
                    controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                    guidance_scale=float(guidance_scale),
                    strength=float(strength),
                    seed=int(seed)
                )
            img_io = io.BytesIO()
            image.save(img_io, 'JPEG', quality=70)
            img_io.seek(0)
            return HttpResponse(img_io, content_type='image/jpeg')
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return HttpResponse('This method is not allowed', status=405)
