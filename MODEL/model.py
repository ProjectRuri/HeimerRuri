import numpy as np
from classModels import ClinicalDataset
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

from util import *





def label_to_int(group: str) -> int:
    """
    tensorflow에서 사용하기 위해선 숫자를 이용해서 처리해야함
    그래서 group은 'CN', 'AD'인 문자열을 사용하기에 매핑한것
    """
    return {'CN': 0, 'AD': 1}[group]  # 예시

def build_tensorflow_dataset(dataset: list[ClinicalDataset]):
    mri_data = []
    labels = []

    for sample in dataset:
        mri_data.append(sample.volume)
        labels.append(label_to_int(sample.label.group))  # 문자열 라벨 → 정수 라벨

        
    mri_data = np.array(mri_data).astype(np.float32)[..., np.newaxis]  # (N, D, H, W, 1)
    labels = np.array(labels).astype(np.int32)

    # TensorFlow Dataset 생성
    return tf.data.Dataset.from_tensor_slices(((mri_data), labels))


def build_model(size:int):
    mri_input = Input(shape=(size, size, size, 1), name='mri_input')
    x = layers.Conv3D(16, kernel_size=3, activation='relu')(mri_input)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(64, kernel_size=3, activation='relu')(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)  # 이진 분류

   

    model = Model(inputs=mri_input, outputs=output)
    # model.summary()
    return model



def build(preprocessed: list[ClinicalDataset], size:int):
    """
    모델 초기 학습을 진행

    INPUT:
        전처리된 ClinicalDataset 리스트
    OUTPUT:
        학습된 모델
    """
    timer("모델 처리 시작")

    # 데이터셋 구성

    # 전처리된 데이터셋을 텐서플로우 데이터셋으로 변환
    tf_dataset = build_tensorflow_dataset(preprocessed)

    # shuffle(100) -> 데이터 순서를 섞어 과적합을 방지
    # batch(8) -> 한번에 8개의 샘플을 묶어 학습(미니 배치 학습)
    # prefetch(tf.data.AUTOTUNE) -> CPU와 GPU의 병렬처리로 속도 향상상

    train_dataset = tf_dataset.shuffle(100).batch(2).prefetch(tf.data.AUTOTUNE)

    model = build_model(size)
    

    # optimizer='adam' -> 경사하강 최적화 알고리즘을 사용
    # loss = 'binary_crossentropy' -> 이진 분류 문제에 적합한 손실 함수
    # metrics=['accuracy'] -> 학습중 정확도를 표기
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_dataset, epochs=10, verbose=2)

    timer("모델 처리 완료")
    return model
