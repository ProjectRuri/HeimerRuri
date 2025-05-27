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



def view_history(history):
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