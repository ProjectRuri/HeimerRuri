from typing import Optional, Dict, Any
import numpy as np

def loader():
    """
    모든 기본 데이터를 갖춘 벡터를 리턴합니다.

    INPUT
        X
    
    OUTPUT
        dataset list
    
    dataset

    """



def load_raw_mri():
    """
    기본적인 원본 mri를 가져오는 함수
    """

    return 0

def load_labels():
    """
    데이터의 정답을 가져오는 함수
    """
    return 0


class ClinicalDataset:
    """
    학습을 위한 단일 샘플 클래스

    설명:
        volume (np.ndarray): 3D MRI 볼륨 데이터
        label (Any): 정답 라벨 (예: 'AD', 0, 1 등)
        modalities (Dict[str, Any]): 임상 정보, 유전자 정보 등 다양한 모달 입력
    """
    
    def __init__(self, volume: np.ndarray, label: Any, modalities: Optional[Dict[str, Any]] = None):
        self.volume = volume              
        self.label = label       
        self.modalities = modalities or {}

    
    def to_tensor_dict(self):
        """PyTorch나 TensorFlow 학습에 적합한 딕셔너리 형태로 변환"""
        return {
            "mri": self.volume,
            "label": self.label,
            "modalities": self.modalities
        }

# raw_clinical_input