import torch
from PIL import Image
import torchvision.transforms as T

class Combine:
    def combine(self, image_paths):
        if len(image_paths) < 2:
            raise ValueError("적어도 두 개 이상의 이미지 경로가 필요합니다.")


        tensors = [T.ToTensor()(Image.open(f"./sp/{path}")) for path in image_paths]


        min_height = min(t.shape[1] for t in tensors)
        min_width = min(t.shape[2] for t in tensors)
        resize = T.Resize(size=(min_height, min_width))
        tensors = [resize(t) for t in tensors]


        combined = sum(tensors)
        combined = torch.clamp(combined, 0, 1)

        return combined




#result_img = T.ToPILImage()(result)
#result_img.save('combined.png')
