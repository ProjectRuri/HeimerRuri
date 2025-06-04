import random
import numpy as np
from classModels import ClinicalDataset
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, regularizers, Input, Model

from sklearn.model_selection import train_test_split,StratifiedKFold
from tqdm.keras import TqdmCallback

from tensorflow.keras.callbacks import EarlyStopping


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




def build_model(size: int) -> Model:
    """
    개선 모델 L2정규화 적용
    L2 : 가중치가 커질수록 loss에 패널티를 부여해서 과의존 현상 방지
    """
    mri_input = Input(shape=(size, size, size, 1), name='mri_input')

    x = layers.Conv3D(16, kernel_size=3, padding='same')(mri_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(32, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling3D()(x)

    # L2 적용
    x = layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(
        1, 
        activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.001)  # 마지막에도 L2 약하게 적용
    )(x)

    model = Model(inputs=mri_input, outputs=output)
    return model



# GPU VRAM에 영향을 주는 요소는 batch size와 모델의 크기

def build(preprocessed: list[ClinicalDataset], size: int, CNcount: int, ADcount: int, use_kfold=False, k=5):
    import numpy as np
    import random
    from sklearn.model_selection import train_test_split, StratifiedKFold
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

   

    # 데이터 제너레이터 함수: volume과 label을 생성
    def generator(preprocessed_list):
        for sample in preprocessed_list:
            volume = np.load(sample.volume_path)
            label = 1 if sample.label.group == "AD" else 0
            yield volume, label

    # TensorFlow Dataset 생성 함수
    def build_tensorflow_dataset(preprocessed_list):
        input_shape = (size, size, size, 1)
        return tf.data.Dataset.from_generator(
            lambda: generator(preprocessed_list),
            output_signature=(
                tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

    # 모델 컴파일 및 학습 수행 함수
    def compile_and_train(model, train_data, val_data):
        # 학습용 및 검증용 데이터셋 구성
        train_dataset = build_tensorflow_dataset(train_data).shuffle(100).batch(4).prefetch(tf.data.AUTOTUNE)
        val_dataset = build_tensorflow_dataset(val_data).batch(4).prefetch(tf.data.AUTOTUNE)

        optimizer = Adam(learning_rate=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        min_early_stop = MinEpochEarlyStopping(min_epoch=10, monitor='val_loss', patience=5, restore_best_weights=True)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # 모델 학습
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            callbacks=[min_early_stop],
            verbose=1
        )
        return model, history

    timer("모델 처리 시작")

    # 클래스 수를 균형 맞추기 위해 최소 수에 맞춤
    min_count = min(CNcount, ADcount)
    cn_data = [x for x in preprocessed if x.label.group == 'CN']
    ad_data = [x for x in preprocessed if x.label.group == 'AD']
    random.shuffle(cn_data) # 정상 데이터중 셔플링
    random.shuffle(ad_data) # 치매 데이터중 셔플링
    balanced_data = cn_data[:min_count] + ad_data[:min_count]
    random.shuffle(balanced_data)   # 치매, 정상 중 최대 갯수에 맞춰 합친 데이터 셔플링
    labels = [1 if x.label.group == 'AD' else 0 for x in balanced_data] # 기존에 'AD', 'CN'과 같이 String으로 있던 데이터를 'AD' => 1, 'CN' => 0 으로 변환. 텐서플로우는 STRING자료형 사용 X

    # K-Fold 교차검증 시작
    if use_kfold:
        print(f"K-Fold ({k}) 교차검증 수행 중...")
        kfold_time = time.time()
        best_model = None
        best_history = None
        best_auc = -1
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(balanced_data, labels)):
            print(f"\n--- Fold {fold+1} ---")
            train_data = [balanced_data[i] for i in train_idx]
            val_data = [balanced_data[i] for i in val_idx]

            model = build_model(size)
            model, history = compile_and_train(model, train_data, val_data)

            # 현재 fold의 AUC 측정
            val_dataset = build_tensorflow_dataset(val_data).batch(4)
            predictions = model.predict(val_dataset).flatten()
            val_labels = np.array([1 if x.label.group == 'AD' else 0 for x in val_data])
            auc = tf.keras.metrics.AUC()(val_labels, predictions).numpy()

            # 최상의 AUC를 기록한 모델과 히스토리 저장
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_history = history

        kfold_time = kfold_time-time.time()
        print(f"K-Fold 교차검증 수행 완료, 소요시간 : {kfold_time:.2f}")
        return best_model, best_history

    else:
        # 단일 학습 모드
        print("단일 학습 실행 중...")
        train_data, val_data = train_test_split(balanced_data, test_size=0.2, stratify=labels)
        model = build_model(size)
        model, history = compile_and_train(model, train_data, val_data)
        timer("단일 학습 완료")
        return model, history







    min_count = min(CNcount, ADcount)
    cn_data = [x for x in preprocessed if x.label.group == 'CN']
    ad_data = [x for x in preprocessed if x.label.group == 'AD']

    random.shuffle(cn_data)
    random.shuffle(ad_data)

    balanced_data = cn_data[:min_count] + ad_data[:min_count]
    random.shuffle(balanced_data)

    # 지도학습 과정에서 학습용이랑 테스트용 분리
    train_data, val_data = train_test_split(balanced_data, test_size=0.2, stratify=[x.label.group for x in balanced_data])



    cn_count, ad_count= 0,0


    # 데이터 검토
    for i in train_data:
        if i.label.group == 'CN':
            cn_count +=1
        else:
            ad_count+=1
    print(f"train data CN : {cn_count}, AD : {ad_count}")

    cn_count, ad_count= 0,0

    for i in val_data:
        if i.label.group == 'CN':
            cn_count +=1
        else:
            ad_count+=1
    print(f"val data CN : {cn_count}, AD : {ad_count}")


    input_shape = (size, size, size, 1)


    
    

    # 데이터셋 구성

    # shuffle(100) -> 데이터 순서를 섞어 과적합을 방지
    # batch(8) -> 한번에 8개의 샘플을 묶어 학습(미니 배치 학습)
    # prefetch(tf.data.AUTOTUNE) -> CPU와 GPU의 병렬처리로 속도 향상

    # 전처리된 데이터셋을 텐서플로우 데이터셋으로 변환
    train_dataset = build_tensorflow_dataset(train_data).shuffle(100).batch(4).prefetch(tf.data.AUTOTUNE)
    val_dataset = build_tensorflow_dataset(val_data).batch(4).prefetch(tf.data.AUTOTUNE)



    # 모델 생성
    model = build_model(size)
    

    # optimizer='adam' -> 경사하강 최적화 알고리즘을 사용
    optimizer = Adam(learning_rate=0.0001)
    
    # 조기 중단할 기준 설정정
    early_stop = EarlyStopping(
        monitor='val_loss',             # 검토할 성능 지표 : 검증 손실
        patience=5,                     # n 번 동안 개선없을시 정지
        restore_best_weights=True       # 성능이 가장 좋았던 시점의 가중치를 복원할건지?
        )
    

    # loss = 'binary_crossentropy' -> 이진 분류 문제에 적합한 손실 함수
    # metrics=['accuracy'] -> 학습중 정확도를 표기
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


    
    print(f"모델 생성에 사용된 데이터 (정상 : {min_count}/{CNcount}, 치매 {min_count}/{ADcount})")

    # 조기 종료 및 최소 학습 지정
    min_early_stop = MinEpochEarlyStopping(min_epoch=10, early_stopping=early_stop)


    # 지도학습
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,  # ← val_loss 계산을 위한 설정 (임시로 train 그대로 사용)
        epochs=50,
        callbacks=[min_early_stop],
        verbose=1
    )

    timer("모델 처리 완료")
    return model, history




class MinEpochEarlyStopping(tf.keras.callbacks.Callback):
    """
    최소 에폭 이후에만 조기 종료를 허용하는 경량 콜백
    """
    def __init__(self, min_epoch, monitor='val_loss', patience=5, restore_best_weights=True):
        super().__init__()
        self.min_epoch = min_epoch
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf')
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if epoch + 1 < self.min_epoch:
            return  # 최소 에폭 전에는 종료 판단 안 함

        if current < self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch}")