import json

from django.http import HttpResponse
from PIL import Image
import qrcode
from django.views.decorators.csrf import csrf_exempt
import io

# 确保在myapp目录中创建了以下导入的模块
from .image_generator import inference

@csrf_exempt  # 允许跨站请求
def generate_image(request):
    if request.method == "POST":
        # 尝试解析请求体中的JSON数据
        data = json.loads(request.body)
        # 获取POST请求参数
        qr_code_content = data.get('qr_code_content', '')
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')

        # 调用之前定义的inference函数进行图像生成
        # 注意：这里省略了inference函数的实现细节，你需要根据前面的描述添加它
        image = inference(
            qr_code_content=qr_code_content,
            prompt=prompt,
            negative_prompt=negative_prompt,
            # 其他参数根据需要添加
        )

        # 将生成的PIL图像转换为字节流以便返回
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        return HttpResponse(img_io, content_type='image/jpeg')
    else:
        return HttpResponse('This method is not allowed', status=405)
