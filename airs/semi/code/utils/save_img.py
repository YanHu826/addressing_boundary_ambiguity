import os
from torchvision import transforms

def save_img(x, suffix):
    img = x.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img_save_dir = os.environ.get('AIRS_SEMI_RESULT_DIR', './result')
    os.makedirs(img_save_dir, exist_ok=True)
    suffix = suffix.replace('jpg', 'png')
    img.save(os.path.join(img_save_dir, suffix))
