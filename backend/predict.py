import io
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = 16_000_000
FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "PPM"}
def decode_image(data):
    if not data:
        raise ValueError("The image is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in FORMATS:
                    raise ValueError("Supported formats: JPEG, PNG, WebP, BMP and PPM.")
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Cannot decode this image, or its dimensions exceed the safety limit.") from exc

def image_array(image):
    # Raw 0..255 floats: the model's Rescaling layer normalizes exactly once.
    return np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)

def locate_sign(image):
    hsv = np.asarray(image.convert("HSV"))
    saturated = (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 45)
    labels, count = ndimage.label(saturated, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return image

    candidates = []
    image_area = image.width * image.height
    for label, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        y_slice, x_slice = region
        width = x_slice.stop - x_slice.start
        height = y_slice.stop - y_slice.start
        box_area = width * height
        if min(width, height) < 16 or box_area < image_area * 0.001:
            continue
        ratio = width / height
        if not 0.55 <= ratio <= 1.6 or box_area > image_area * 0.7:
            continue
        component_area = int(np.count_nonzero(labels[region] == label))
        if component_area / box_area < 0.12:
            continue
        center_x = (x_slice.start + x_slice.stop) / 2
        center_y = (y_slice.start + y_slice.stop) / 2
        offset = np.hypot(
            (center_x - image.width / 2) / image.width,
            (center_y - image.height / 2) / image.height,
        )
        candidates.append((component_area / (1 + offset), x_slice, y_slice))

    if not candidates:
        return image
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] >= candidates[0][0] * 0.7:
        return image

    _, x_slice, y_slice = candidates[0]
    width = x_slice.stop - x_slice.start
    height = y_slice.stop - y_slice.start
    side = min(max(width, height) * 1.35, image.width, image.height)
    center_x = (x_slice.start + x_slice.stop) / 2
    center_y = (y_slice.start + y_slice.stop) / 2
    left = int(np.clip(center_x - side / 2, 0, image.width - side))
    top = int(np.clip(center_y - side / 2, 0, image.height - side))
    return image.crop((left, top, left + int(side), top + int(side)))

def inference_array(image):
    return image_array(locate_sign(image))
