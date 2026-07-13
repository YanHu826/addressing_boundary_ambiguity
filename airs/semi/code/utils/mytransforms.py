import numbers
import random

import cv2
import numpy as np
import scipy.ndimage
import torchvision.transforms.functional as F
from PIL import Image

try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    _RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _RESAMPLE_BILINEAR = Image.BILINEAR
    _RESAMPLE_NEAREST = Image.NEAREST


def _label_keys(data):
    return [key for key in data.keys() if key.startswith('label')]


def _map_label_values(data, fn):
    updated = dict(data)
    for key in _label_keys(data):
        updated[key] = fn(data[key])
    return updated


class ToTensor(object):

    def __call__(self, data):
        updated = _map_label_values(data, F.to_tensor)
        updated['image'] = F.to_tensor(data['image'])
        return updated


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        updated = _map_label_values(
            data,
            lambda label: F.resize(label, self.size, interpolation=_RESAMPLE_NEAREST),
        )
        updated['image'] = F.resize(data['image'], self.size, interpolation=_RESAMPLE_BILINEAR)
        return updated


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            updated = _map_label_values(data, F.hflip)
            updated['image'] = F.hflip(data['image'])
            return updated

        return data


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            updated = _map_label_values(data, F.vflip)
            updated['image'] = F.vflip(data['image'])
            return updated

        return data


class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, data):

        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            updated = _map_label_values(
                data,
                lambda label: F.rotate(
                    label,
                    angle,
                    interpolation=_RESAMPLE_NEAREST,
                    expand=self.expand,
                    center=self.center,
                ),
            )
            updated['image'] = F.rotate(
                    data['image'],
                    angle,
                    interpolation=_RESAMPLE_BILINEAR,
                    expand=self.expand,
                    center=self.center,
                )
            return updated

        return data


class RandomZoom(object):
    def __init__(self, zoom=(0.8, 1.2)):
        self.min, self.max = zoom[0], zoom[1]

    def __call__(self, data):
        if random.random() < 0.5:
            image = data['image']
            image = np.array(image)

            zoom = random.uniform(self.min, self.max)
            zoom_image = clipped_zoom(image, zoom, order=1)

            zoom_image = Image.fromarray(zoom_image.astype('uint8'), 'RGB')
            updated = dict(data)
            updated['image'] = zoom_image
            for key in _label_keys(data):
                zoom_label = clipped_zoom(np.array(data[key]), zoom, order=0)
                updated[key] = Image.fromarray(zoom_label.astype('uint8'), 'L')
            return updated

        return data


class ImageRandomZoom(object):
    def __init__(self, zoom=(0.8, 1.2)):
        self.min, self.max = zoom[0], zoom[1]

    def __call__(self, image):
        if random.random() < 0.5:
            zoom = random.uniform(self.min, self.max)
            zoomed = clipped_zoom(np.array(image), zoom, order=1)
            return Image.fromarray(zoomed.astype('uint8'), mode=image.mode)
        return image


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        zoom_in = scipy.ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `zoom_in` might still be slightly different with `img` due to rounding, so
        # trim off any extra pixels at the edges or zero-padding

        if zoom_in.shape[0] >= h:
            zoom_top = (zoom_in.shape[0] - h) // 2
            sh = h
            out_top = 0
            oh = h
        else:
            zoom_top = 0
            sh = zoom_in.shape[0]
            out_top = (h - zoom_in.shape[0]) // 2
            oh = zoom_in.shape[0]
        if zoom_in.shape[1] >= w:
            zoom_left = (zoom_in.shape[1] - w) // 2
            sw = w
            out_left = 0
            ow = w
        else:
            zoom_left = 0
            sw = zoom_in.shape[1]
            out_left = (w - zoom_in.shape[1]) // 2
            ow = zoom_in.shape[1]

        out = np.zeros_like(img)
        out[out_top:out_top + oh, out_left:out_left + ow] = zoom_in[zoom_top:zoom_top + sh, zoom_left:zoom_left + sw]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


class Translation(object):
    def __init__(self, translation):
        self.translation = translation

    def __call__(self, data):
        if random.random() < 0.5:
            image = data['image']
            image = np.array(image)
            rows, cols, ch = image.shape

            translation = random.uniform(0, self.translation)
            tr_x = translation / 2
            tr_y = translation / 2
            Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

            translate_image = cv2.warpAffine(
                image,
                Trans_M,
                (cols, rows),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            translate_image = Image.fromarray(translate_image.astype('uint8'), 'RGB')
            updated = dict(data)
            updated['image'] = translate_image
            for key in _label_keys(data):
                translate_label = cv2.warpAffine(
                    np.array(data[key]),
                    Trans_M,
                    (cols, rows),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                updated[key] = Image.fromarray(translate_label.astype('uint8'), 'L')
            return updated

        return data


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        # i = torch.randint(0, h - th + 1, size=(1, )).item()
        # j = torch.randint(0, w - tw + 1, size=(1, )).item()
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, data):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        img = data['image']
        updated = dict(data)
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            for key in _label_keys(updated):
                updated[key] = F.pad(updated[key], self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            for key in _label_keys(updated):
                updated[key] = F.pad(updated[key], (self.size[1] - updated[key].size[0], 0), self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            for key in _label_keys(updated):
                updated[key] = F.pad(updated[key], (0, self.size[0] - updated[key].size[1]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        for key in _label_keys(updated):
            updated[key] = F.crop(updated[key], i, j, h, w)
        updated['image'] = img
        return updated


class AnatomyAwareRandomCrop(RandomCrop):
    """Random crop with a probabilistic foreground anchor.

    Most crops stay anatomically informative (important for small structures
    such as PS), while the fallback random branch preserves background context.
    """

    def __init__(
        self,
        size,
        focus_keys=('label',),
        focus_prob=0.6,
        min_positive_pixels=1,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode='constant',
    ):
        super().__init__(
            size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
        )
        self.focus_keys = tuple(focus_keys)
        self.focus_prob = float(focus_prob)
        self.min_positive_pixels = int(min_positive_pixels)

    @staticmethod
    def _positive_coords(label):
        arr = np.array(label)
        if arr.ndim > 2:
            arr = arr[..., 0]
        return np.where(arr > 0)

    def _get_focus_params(self, img, data):
        if random.random() >= self.focus_prob:
            return None

        h, w = img.size[1], img.size[0]
        th, tw = self.size
        if w == tw and h == th:
            return 0, 0, h, w
        if h < th or w < tw:
            return None

        for key in self.focus_keys:
            label = data.get(key, None)
            if label is None:
                continue
            ys, xs = self._positive_coords(label)
            if len(ys) < self.min_positive_pixels:
                continue
            idx = random.randrange(len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            top_min = max(0, cy - th + 1)
            top_max = min(cy, h - th)
            left_min = max(0, cx - tw + 1)
            left_max = min(cx, w - tw)
            if top_min <= top_max and left_min <= left_max:
                return (
                    random.randint(top_min, top_max),
                    random.randint(left_min, left_max),
                    th,
                    tw,
                )
        return None

    def __call__(self, data):
        img = data['image']
        updated = dict(data)
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            for key in _label_keys(updated):
                updated[key] = F.pad(updated[key], self.padding, self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            for key in _label_keys(updated):
                updated[key] = F.pad(updated[key], (self.size[1] - updated[key].size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            for key in _label_keys(updated):
                updated[key] = F.pad(updated[key], (0, self.size[0] - updated[key].size[1]), self.fill, self.padding_mode)

        params = self._get_focus_params(img, updated)
        if params is None:
            params = self.get_params(img, self.size)
        i, j, h, w = params
        updated['image'] = F.crop(img, i, j, h, w)
        for key in _label_keys(updated):
            updated[key] = F.crop(updated[key], i, j, h, w)
        return updated


class Normalization(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image = F.normalize(image, self.mean, self.std)
        updated = dict(sample)
        updated['image'] = image
        return updated
