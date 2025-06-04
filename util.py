def ask_yes_no(prompt: str, default: str = 'y') -> bool:
    """
    선택지를 묻는 함수
    
    INPUT:
        prompt : 물어볼 내용
        default : 공백 입력시 사용할 답안
    OUTPUT:
        bool : 결과
    """


    # 입력 대소문자 통일
    default = default.lower()
    if default not in ['y', 'n']:
        raise ValueError("default must be 'y' or 'n'")

    # 공백시 사용할 답안 처리
    prompt_suffix = "((y)/n)" if default == 'y' else "(y/(n))"


    # 정상적인 답을 입력할때 까지 입력 요청
    while True:
        answer = input(f"{prompt} {prompt_suffix}: ").strip().lower()
        if answer == '':
            return default == 'y'
        elif answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Please enter y or n.")



import os
import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import psutil

from classModels import ClinicalDataset

def timer(label="Timer"):
    """
    이전에 이 힘수를 호출한 시점으로부터 시간을 출력
    INPUT:
        label : 설명
    """
    if not hasattr(timer, "_last_time"):
        timer._last_time = time.time()
        print(f"\n[{label}] 시작\n")
    else:
        now = time.time()
        elapsed = now - timer._last_time
        print(f"\n[{label}] 경과 시간: {elapsed:.2f}초\n")
        timer._last_time = now



def reset_log(file_path="log.txt"):
    """
    로그 파일 초기화 하는 함수
    INPUT:
        file_path: 로그 파일 이름 입력 없을시 기본 log.txt로 작성됨됨

    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("")  # 빈 문자열로 덮어쓰기


def print_and_log(text, file_path="log.txt"):
    """
    로그 찍어주는 함수
    INPUT:
        text : 찍을 로그
        file_path : 로그 파일 이름 입력 없을시 기본 log.txt로 작성됨
    """
   
    text = str(text)
    #print(text)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")



def test_memory_load(samples:list[ClinicalDataset], verbose=True):
    """
    데이터 메모리 사용량 확인용 코드
    INPUT:
        samples : 테스트할 데이터(list[ClinicalDataset])
        verbose : 중간과정 출력 여부
    OUTPUT:
        X
        
    """
    volumes = []
    for i, sample in enumerate(samples):
        vol = np.load(sample.volume_path).astype(np.float32)
        volumes.append(vol)
        if verbose and i % 10 == 0:
            print(f"{i+1}개 로드 완료")
    print(f"총 {len(volumes)}개 샘플 메모리에 적재 완료")
    test_volumes = volumes

    process = psutil.Process(os.getpid())
    print(f"현재 메모리 사용량: {process.memory_info().rss / 1024 ** 2:.2f} MB")



def plot_history(history):
    # 정확도 시각화
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 손실값 시각화
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_prediction_histogram(labels, predictions, threshold=0.5):
    """
    예측 점수 히스토그램을 시각화합니다.
    
    Parameters:
    - labels: 실제 정답 (리스트나 배열), 예: ["AD", "CN", ...]
    - predictions: 모델 예측 점수 (0~1 사이), 예: [0.83, 0.41, ...]
    - threshold: 분류 임계값 (기본값은 0.5)
    """
    df = pd.DataFrame({
        "Label": labels,
        "Prediction": predictions
    })

    ad_scores = df[df["Label"] == "AD"]["Prediction"]
    cn_scores = df[df["Label"] == "CN"]["Prediction"]

    plt.figure(figsize=(8, 5))
    plt.hist(ad_scores, bins=20, alpha=0.7, label="AD", color='red')
    plt.hist(cn_scores, bins=20, alpha=0.7, label="CN", color='blue')
    plt.axvline(x=threshold, color='black', linestyle='--', label=f"Threshold = {threshold}")
    plt.title("Prediction Score Histogram")
    plt.xlabel("Prediction Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

def plot_roc_curve(labels, predicts):
    """
    모델 분류 성능 표시 그래프
    좌상단에 그래프가 모일수록 성능 좋음
    """
    y_true = [0 if lbl == "CN" else 1 for lbl in labels]

    fpr, tpr, thresholds = roc_curve(y_true, predicts)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_all_metrics(history, labels, predictions, threshold=0.5):
    """
    데이터 종합 시각화 방식
    텐서플로우의 history 2개
    roc_curve <- 모델 분류 성능을 표시 -> 좌상단에 그래프가 쏠릴수록 성능 높음
    prediction histogram <- 예측 결과 분포 표시, 
    """

    y_true = [0 if lbl == "CN" else 1 for lbl in labels]
    fpr, tpr, _ = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, 
                  tpr)
    labels = np.ravel(labels)
    predictions = np.ravel(predictions)
    
    df = pd.DataFrame({"Label": labels, "Prediction": predictions})
    ad_scores = df[df["Label"] == "AD"]["Prediction"]
    cn_scores = df[df["Label"] == "CN"]["Prediction"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 텐서 플로우 학습 epoch별로 정확도 그래프
    axs[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axs[0, 0].set_title('Accuracy over Epochs')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()

    # 텐서 플로우 학습 epoch별로 손실 그래프
    axs[0, 1].plot(history.history['loss'], label='Train Loss')
    axs[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axs[0, 1].set_title('Loss over Epochs')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # ROC Curve
    axs[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axs[1, 0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axs[1, 0].set_xlabel('False Positive Rate')
    axs[1, 0].set_ylabel('True Positive Rate')
    axs[1, 0].set_title('ROC Curve')
    axs[1, 0].legend(loc='lower right')

    # 예측값 분포 그래프
    axs[1, 1].hist(ad_scores, bins=20, alpha=0.7, label="AD", color='red')
    axs[1, 1].hist(cn_scores, bins=20, alpha=0.7, label="CN", color='blue')
    axs[1, 1].axvline(x=threshold, color='black', linestyle='--', label=f"Threshold = {threshold}")
    axs[1, 1].set_title("Prediction Score Histogram")
    axs[1, 1].set_xlabel("Prediction Score")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
