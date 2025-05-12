
import numpy as np
from scipy.ndimage import zoom
from classModels import *

def resize_volume(volume, target_shape=(128, 128, 128)):
    """
    3D MRI 볼륨을 target_shape로 리사이즈함

    Args:
        volume (np.ndarray): 원본 MRI 볼륨 (D, H, W)
        target_shape (tuple): 원하는 출력 크기 (D, H, W)

    Returns:
        np.ndarray: 리사이즈된 MRI 볼륨
    """
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, zoom=zoom_factors, order=1)  # 선형 보간
    return resized.astype(np.float32)



def preprocess(origin_data: list[ClinicalDataset]):
    """
    전처리를 진행
    INPUT:
        loader()을 통해 출력된 origin_data
        즉 ClinicalDataset 리스트 혹은 벡터
    OUTPUT:
        전처리가 진행된
    """
    for i in origin_data:
        resized = resize_volume(i.volume)
        i.volume = resized
    print("전처리 완료")
    return origin_data