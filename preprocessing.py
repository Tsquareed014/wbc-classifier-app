
import numpy as np

def preprocess_image(image, width=128, height=128, normalization="0-1"):
    """
    Resize and normalize a PIL image.

    Args:
      image (PIL.Image): input image.
      width (int): target width.
      height (int): target height.
      normalization (str): "0-1" or "mean-std".

    Returns:
      image_array: np.ndarray shape (1, height, width, 3)
      resized PIL image
    """
    try:
        # resize
        image = image.resize((width, height))
        arr = np.array(image).astype(np.float32)

        # normalization
        if normalization == "0-1":
            arr /= 255.0
        else:
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)

        # add batch dim
        image_array = np.expand_dims(arr, axis=0)
        return image_array, image

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

