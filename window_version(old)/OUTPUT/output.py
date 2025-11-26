import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def plot_average_feature_maps(model, ad_tensors, cn_tensors, layer_name=None, slice_idx=None, max_filters=8):
    """
    AD vs CN 평균 feature map 비교 시각화
    :param model: 학습된 keras 모델
    :param ad_tensors: AD 입력 텐서 리스트 [(1, D, H, W, 1)]
    :param cn_tensors: CN 입력 텐서 리스트 [(1, D, H, W, 1)]
    :param layer_name: 대상 Conv 레이어 이름
    :param slice_idx: 시각화할 slice 인덱스
    :param max_filters: 최대 시각화할 필터 수
    """
    from tensorflow.keras.models import Model
    import matplotlib.pyplot as plt
    import numpy as np

    if layer_name is None:
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break

    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    def get_avg_feature(tensor_list):
        total = None
        for t in tensor_list:
            feat = intermediate_model.predict(t)[0]  # (D, H, W, C)
            if total is None:
                total = feat
            else:
                total += feat
        return total / len(tensor_list)

    avg_ad = get_avg_feature(ad_tensors)
    avg_cn = get_avg_feature(cn_tensors)

    if slice_idx is None:
        slice_idx = avg_ad.shape[0] // 2

    num_filters = avg_ad.shape[-1]
    filters_to_show = min(num_filters, max_filters)

    plt.figure(figsize=(3 * filters_to_show, 6))
    for i in range(filters_to_show):
        plt.subplot(2, filters_to_show, i + 1)
        plt.imshow(avg_cn[slice_idx, :, :, i], cmap='viridis')
        plt.title(f"CN Filter {i}")
        plt.axis('off')

        plt.subplot(2, filters_to_show, filters_to_show + i + 1)
        plt.imshow(avg_ad[slice_idx, :, :, i], cmap='viridis')
        plt.title(f"AD Filter {i}")
        plt.axis('off')

    plt.suptitle(f"AD vs CN 평균 Feature Map (Slice {slice_idx})", fontsize=16)
    plt.tight_layout()
    plt.show()

def output():
    """
    진행한 결과를 출력
    ex) 정확도, 틀린 데이터 목록, 학습에 사용된 데이터 목록
    """

    
    return