
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

    



    timer("전처리 완료")
    return origin_data