
import gc
import numpy as np
from scipy.ndimage import zoom
from classModels import *
from tqdm import tqdm
from util import *







def normalization(volume, clip_percentile=0.2, replace_value=None):
    lower_bound = np.percentile(volume, clip_percentile)
    mask = volume > lower_bound

    mean = np.mean(volume[mask])
    std = np.std(volume[mask])
    std = max(std, 1e-6)

    volume[mask] = (volume[mask] - mean) / std
    if replace_value is None:
        replace_value = np.min(volume[mask])
    volume[~mask] = replace_value
    return volume

def preprocess(origin_data: list[ClinicalDataset], size: int) -> list[ClinicalDataset]:
    """
    전처리를 진행
    INPUT:
        loader()을 통해 출력된 origin_data (ClinicalDataset 리스트)
    OUTPUT:
        전처리된 ClinicalDataset 리스트
    """
    timer("전처리 시작")

    for sample in tqdm(origin_data, desc="정규화 진행 중"):
        volume = sample.load_volume()




        volume = normalization(volume)

        volume[np.abs(volume) < 0.2] = 0
        


        # 정규화된 볼륨 저장
        save_path = sample.volume_path.parent / f"normalized_{sample.volume_path.name}"
        np.save(save_path, volume.astype(np.float32))

        # 경로 업데이트
        sample.volume_path = save_path

    timer("전처리 완료")
    return origin_data
