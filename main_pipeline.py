import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# TF_CPP_MIN_LOG_LEVEL 값:

# '0': 모든 메시지 출력 (기본값)

# '1': INFO 메시지 숨김

# '2': INFO + WARNING 숨김 (⚠️ 추천)

# '3': INFO + WARNING + ERROR 숨김 (주의 필요)

from LOADER.loader import *
from PREPROCESS.preprocess import *
from MODEL.model import *
from OUTPUT.output import *
from classModels import *
from util import *



from tensorflow.keras.utils import plot_model


def main():
    # ─────터미널 입력───────────────────────────────
    # cd 이용해서 heimerruri로 이동
    # 가상환경 설정 안했다면 아래 들여쓴 코드 실행
    # 아나콘다3을 사용중임, python버전은 3.10 사용
    # conda create -n tf_gpu python=3.10
    # conda activate tf_gpu

    # 윈도우는 텐서플로우에서 gpu를 사용하기 위해서는 tensorflow 2.10을 사용해야함
    # 이에 호환되는 cuda와 cudnn을 매칭시켜야함
    # cudnn - 8.1 -> 여러 버전이 같이 있으니 주의!
    # https://developer.nvidia.com/rdp/cudnn-archive#a-collapse81-112
    # cuda - 11.2
    # https://developer.nvidia.com/cuda-11.2.2-download-archive
    # 데이터 시각화 툴 (Graphviz 12.2.1)
    # https://graphviz.org/download/

    # 텐서 플로우 버전을 맞춰야함
    # pip install tensorflow==2.10

    # 사용중인 패키지
    # pip install scipy pandas matplotlib nibabel pydot tqdm


    # ─────데이터 준비───────────────────────────────

    # 이미 학습이 진행된경우 바로 모델 학습으로 이동해도됨


    # 작업에 필요한 선택지 선입력
    dcm_to_nii_process = ask_yes_no("DCM 변환이 필요합니까?", default='n')
    model_visualization = ask_yes_no("모델 시각화가 필요합니까?", default='n')

    size = 64


    
    timer("프로그램 시작")
    
    # 초기 데이터 로드
    origin_data = loader(dcm_to_nii_process)
    
    # 전처리
    preprocessed = preprocess(origin_data,size)
    
    # 모델 처리
    fit_model = build(preprocessed,size)
    

    # view_volume(sample) # 입력한 데이터 시각으로 확인
    # 모델 시각화
    if(model_visualization):
        plot_model(fit_model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)



    # 입력한 데이터(치매 수, 정상 수)
    ad, cn = 0,0
    # 치매-> a , 정상-> c
    # 데이터는 정상이지만   예측은 치매 -> ca
    # 데이터는 치매지만     예측은 정상 -> ac
    aa, ac, ca, cc = 0,0,0,0
    # 테스트용 샘플
    for i in range(100):
        # 랜덤 샘플 선택
        index = random.randrange(1,len(preprocessed))
        sample = preprocessed[index]  # ClinicalDataset

        # volume을 텐서플로에 넣어둘 규격으로 변경
        input_tensor = np.expand_dims(sample.volume, axis=(0, -1))  # (1, D, H, W, 1)

        # 예측
        prediction = fit_model.predict(input_tensor)
        
        # 라벨 출력
        print(sample.label)
        # CN => 0 AD => 1

        # 결과 처리
        result = (prediction>0.5).astype(int)
        resultStr = ("AD" if result else "CN")

        # sample의 그룹
        nowGroup = sample.label.group


        if nowGroup == "CN":
            cn += 1
        else:
            ad += 1
        # 혼동 행렬 구성
        if nowGroup == "AD" and resultStr == "AD":
            aa += 1
        elif nowGroup == "AD" and resultStr == "CN":
            ac += 1
        elif nowGroup == "CN" and resultStr == "AD":
            ca += 1
        elif nowGroup == "CN" and resultStr == "CN":
            cc += 1



        print(f"예측 결과 : {resultStr}, prediction : {prediction}")

    print(f"치매 : {ad}, 정상 : {cn}, 정확도 : {(aa+cc)/(ad+cn)}")
    print(f"치매->치매 : {aa}, 치매->정상 : {ac}")
    print(f"정상->치매 : {ca}, 정상->정상 : {cc}")
    

    # ─────모델 학습────────────────────────────────

    # 지도학습용 데이터 로드

    # 전처리

    # 지도학습 예시 
    #    for X, Y in dataloader:
    #     optimizer.zero_grad()         # 🔸 이전 gradient를 0으로 초기화
    #     Y_hat = model(X)              # 🔸 forward: 예측값 계산
    #     loss = criterion(Y_hat, Y)    # 🔸 예측 vs 정답 → 손실
    #     loss.backward()               # 🔸 역전파: gradient 계산
    #     optimizer.step()              # 🔸 gradient 기반으로 가중치 갱신

    # 텐서플로우나 파이토치를 사용할 경우 체크포인트를 만들어서 저장할것
    # 저장할 경우 초키 모델 학습이 불필요함

main()