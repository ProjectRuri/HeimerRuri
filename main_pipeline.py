from LOADER.loader import *
from PREPROCESS.preprocess import *
from MODEL.model import *
from OUTPUT.output import *

def main():
    
    # ─────데이터 준비───────────────────────────────

    # 이미 학습이 진행된경우 바로 모델 학습으로 이동해도됨

    # 초기 데이터 로드
    origin_data = loader()

    # 전처리
    preprocessed = preprocess(origin_data)

    # 모델 처리
    fit_model = fit(preprocessed)

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