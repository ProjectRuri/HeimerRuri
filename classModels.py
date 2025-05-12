from typing import Optional, Dict, Any
import numpy as np


class Label:
    """
    라벨 클래스

    설명:
        Subject         : 환자 식별 번호
        Group           : 치매 여부
        Image Data ID   : 촬영 식별 번호
    """
    def __init__(self, subject: str, group: str, imageDataID: str):
        self.subject = subject
        self.group = group
        self.imageDataID = imageDataID


    def __repr__(self):
        return f"(subject = {self.subject}, group = {self.group}, imageDataID = {self.imageDataID})\n"
 

class ClinicalDataset:
    
    """
    학습을 위한 단일 샘플 클래스

    설명:
        volume (np.ndarray): 3D MRI 볼륨 데이터
        label (Any): 정답 라벨 (예: 'AD', 0, 1 등)
        modalities (Dict[str, Any]): 임상 정보, 유전자 정보 등 다양한 모달 입력
    """
    
    def __init__(self, volume: np.ndarray, label: Label, modalities: Optional[Dict[str, Any]] = None):
        self.volume = volume                    # mri 3d 모델? 3차원 배열?          
        self.label = label                      # 라벨
        self.modalities = modalities or {}      # mmse, ldkjflskd~~~~

    
    def to_tensor_dict(self):
        """PyTorch나 TensorFlow 학습에 적합한 딕셔너리 형태로 변환"""
        return {
            "mri": self.volume,
            "label": self.label,
            "modalities": self.modalities
        }
    
    def __repr__(self):
        return f"----\nlabel : {self.label}modalities {self.modalities}\n----\n"
