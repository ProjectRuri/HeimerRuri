import os
import random
import os, psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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


#from tensorflow.keras.utils import plot_model


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
    # pip install scipy pandas matplotlib nibabel pydot tqdm psutil scikit-learn


    # ─────데이터 준비───────────────────────────────

    # 이미 학습이 진행된경우 바로 모델 학습으로 이동해도됨


    # 작업에 필요한 선택지 선입력
    dcm_to_nii_process = ask_yes_no("DCM 변환이 필요합니까?", default='n')
    model_visualization = ask_yes_no("모델 시각화가 필요합니까?", default='n')
    
    size = 64 # 96


    
    timer("프로그램 시작")
    
    # 초기 데이터 로드
    origin_data = loader_parallel_process(dcm_to_nii_process, size,max_workers=16)
    
    # 전처리

    preprocessed = preprocess(origin_data,size)
    
    
    random.shuffle(preprocessed)

    adList, cnList =[],[]

    for i in preprocessed:
        if i.label.group == "CN":
            cnList.append(i)
        else:
            adList.append(i)


    prediction_size = 50

    preprocessed = cnList[prediction_size:] + adList[prediction_size:]

    test_list = cnList[:prediction_size]+adList[:prediction_size]

    print(f"정상 데이터 수 : {len(cnList)}, 치매 데이터 수 : {len(adList)}")

    # 사용 메모리 확인용 코드
    # test_memory_load(test_list)

    

    # 모델 처리
    fit_model, history = build(preprocessed,size,len(cnList)-prediction_size, len(adList)-prediction_size,True,5)
    

    # view_volume(sample) # 입력한 데이터 시각으로 확인
    # 모델 시각화
    if(model_visualization):
        plot_model(fit_model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

    
    


        
    input_tensors = []
    labels = []
    raw_labels = []

    swit = False

    for sample in test_list[:100]:
        volume = sample.load_volume()
        if swit == False:
            # print(np.mean(volume), np.std(volume))
            swit=True
        input_tensor = np.expand_dims(volume, axis=(0, -1))  # (1, D, H, W, 1)
        input_tensors.append(input_tensor[0])  # remove batch dim
        labels.append(sample.label.group)
        raw_labels.append(repr(sample.label))

    input_tensors = np.array(input_tensors)  # (100, D, H, W, 1)

    # 한 번에 예측
    predictions = fit_model.predict(input_tensors)



    # 로그 기록을 위한 로그파일 이름 지정
    prediction_log = "예측 결과.txt"
    reset_log(prediction_log)
    
    
    def confusion_matrix_classification(confusion_matrix,label, prediction):
        if label == "AD":
            confusion_matrix["AD"] +=1
        else:
            confusion_matrix["CN"] +=1
        
        if label == "AD" and prediction == "AD":
            confusion_matrix["AA"] += 1
        elif label == "AD" and prediction == "CN":
            confusion_matrix["AC"] += 1
        elif label == "CN" and prediction == "AD":
            confusion_matrix["CA"] += 1
        elif label == "CN" and prediction == "CN":
            confusion_matrix["CC"] += 1



    threshold_list=[]
    threshold_count = 20
    temp = 1/threshold_count
    for i in range(threshold_count):
        threshold_list.append(temp*i)

    acc_list=[]


    best_threshold = 0
    best_confusion_matrix= {'AD':0,'CN':0,'AA':0,'AC':0,'CA':0,'CC':0}
    best_acc = 0

    # threshold 경계 탐색 적용된 버전
    # 적절한 경계 찾기

    for threshold in threshold_list:
        # 치매-> a , 정상-> c
        # 데이터는 정상이지만   예측은 치매 -> ca
        # 데이터는 치매지만     예측은 정상 -> ac
        confusion_matrix= {'AD':0,'CN':0,'AA':0,'AC':0,'CA':0,'CC':0}

        for i in range(len(predictions)):
            pred = predictions[i]
            result = (pred > threshold).astype(int)
            resultStr = "AD" if result else "CN"

            confusion_matrix_classification(confusion_matrix,labels[i],resultStr)
            

        # 정확도 계산
        acc = (confusion_matrix["AA"]+confusion_matrix["CC"]) / (confusion_matrix["AD"] + confusion_matrix["CN"])
        
        acc_list.append(acc)

        # 기록 경신시 갱신
        if(acc >best_acc):
            best_acc = acc
            best_confusion_matrix = confusion_matrix
            best_threshold = threshold

    
    print(f"치매 : {ad}, 정상 : {cn}, 정확도 : {(aa+cc)/(ad+cn)}")
    print(f"치매->치매 : {aa}, 치매->정상 : {ac}")
    print(f"정상->치매 : {ca}, 정상->정상 : {cc}")



    # cnn futuremap 의미 
    # CNN의 중간 레이어 출력 (feature map)을 직접 추출하여
    # 뇌의 어느 부위에 필터가 반응하는지 확인
    # 향후 AD/CN을 구분하는 데 어떤 특징을 학습하는지 해석 가능
    
    # cnn FutureMap 시각화
    from OUTPUT.output import plot_average_feature_maps #평균내서 비교

    # AD, CN 샘플 각각 여러 개 선택
    ad_tensors = [np.expand_dims(x.volume, axis=(0, -1)) for x in test_list if x.label.group == "AD"][:100]
    cn_tensors = [np.expand_dims(x.volume, axis=(0, -1)) for x in test_list if x.label.group == "CN"][:100]

    plot_average_feature_maps(fit_model, ad_tensors, cn_tensors)


    

    # 대표 한개씩만만
    # ad_sample = next(x for x in test_list if x.label.group == "AD")
    # cn_sample = next(x for x in test_list if x.label.group == "CN")
    # ad_tensor = np.expand_dims(ad_sample.volume, axis=(0, -1))
    # cn_tensor = np.expand_dims(cn_sample.volume, axis=(0, -1))
    # compare_feature_maps(fit_model, ad_tensor, cn_tensor)
    # 최적의 threshold기준으로 log 기록
    for i in range(len(predictions)):
        pred = predictions[i]
        result = (pred > best_threshold).astype(int)
        resultStr = "AD" if result else "CN"
        print_and_log(raw_labels[i], prediction_log)
        print_and_log(f"예측 결과 : {resultStr}, prediction : {pred}", prediction_log)



    # threshold 경계 탐색 최고 기록 출력
    print(f"최고 기록 - threshold : {best_threshold}")
    print(f"치매 : {best_confusion_matrix['AD']}, 정상 : {best_confusion_matrix['CN']}, 정확도 : {best_acc}")
    print(f"치매->치매 : {best_confusion_matrix['AA']}, 치매->정상 : {best_confusion_matrix['AC']}")
    print(f"정상->치매 : {best_confusion_matrix['CA']}, 정상->정상 : {best_confusion_matrix['CC']}")

    for i in range(threshold_count):
        print(f"threshold : {threshold_list[i]:.2f}, acc : {acc_list[i]:.2f}")


    # 시각화
    plot_all_metrics(history,labels, predictions, best_threshold)



if __name__ == "__main__":
    main()
