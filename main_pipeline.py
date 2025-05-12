from LOADER.loader import *
from PREPROCESS.preprocess import *
from MODEL.model import *
from OUTPUT.output import *

from classModels import *
def main():
    # ─────터미널 입력───────────────────────────────
    # cd 이용해서 heimerruri로 이동
    # 가상환경 설정 안했다면 아래 들여쓴 코드 실행
    # conda create -n tf_env python=3.10
    # conda activate tf_env
    # pip install tensorflow

    # 사용중인 패키지
    # pip install scipy pandas matplotlib nibabel

    # venv_tf\Scripts\activate
    

    # ─────데이터 준비───────────────────────────────

    # 이미 학습이 진행된경우 바로 모델 학습으로 이동해도됨

    # 초기 데이터 로드
    origin_data = loader()

    # 전처리
    preprocessed = preprocess(origin_data)

    # 모델 처리
    fit_model = build(preprocessed)

    sample = origin_data[0].volume  # ClinicalDataset.volume
    input_tensor = np.expand_dims(sample, axis=(0, -1))  # (1, D, H, W, 1)

    prediction = fit_model.predict(input_tensor)
    print(origin_data[0].label)
    print("예측 결과:", prediction)


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