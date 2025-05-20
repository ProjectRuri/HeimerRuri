
import gc
import numpy as np
from scipy.ndimage import zoom
from classModels import *
from tqdm import tqdm
from util import *






def preprocess(origin_data: list[ClinicalDataset], size:int)-> list[ClinicalDataset]:
    """
    전처리를 진행
    INPUT:
        loader()을 통해 출력된 origin_data
        즉 ClinicalDataset 리스트 혹은 벡터
    OUTPUT:
        전처리가 진행된
    """
    timer("전처리 시작")

    # target_shape = (size, size, size)
    # resized_dir = Path("resized_volumes")
    # resized_dir.mkdir(exist_ok=True)

    # processed = []

    # for i in tqdm(origin_data, desc="볼륨 리사이즈 중"):
    #     # 원본 볼륨 로드
    #     volume = np.load(i.volume_path)

    #     # 리사이즈
    #     resized = resize_volume(volume, target_shape)

    #     # 경로 생성 및 저장
    #     save_path = resized_dir / f"{i.label.ID}.npy"
    #     np.save(save_path, resized)

    #     # 새 ClinicalDataset 생성 (경로만 포함)
    #     new_item = ClinicalDataset(save_path, i.label)
    #     processed.append(new_item)

    #     # 메모리 해제
    #     del volume, resized
    #     gc.collect()

    timer("전처리 완료")
    return origin_data