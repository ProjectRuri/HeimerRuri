from functools import partial
import subprocess
import concurrent
import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import gc
from typing import Optional, Dict, Any
from matplotlib.widgets import Slider
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
from classModels import Label
from util import *
from scipy.ndimage import zoom



def convert_dcm_to_nii(dicom_dir:Path, output_dir:Path):
    """
    dcm파일을 nii로 변환하는 프로세스

    INPUT:
        dicom_dir(Path) : dcm파일이 있는 경로
        output_dir(Path) : 완성된 nii파일이 저장될 공간
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    LOADER_DIR = Path(__file__).resolve().parent
    dcm2niixLink = LOADER_DIR/"TOOL"/"dcm2niix.exe"

    command = [
        str(dcm2niixLink),
        "-z", "y",
        "-o", str(output_dir),
        str(dicom_dir)
    ]
    result = subprocess.run(command,capture_output=True, text=True)

    if result.returncode == 0:
        
        # print("변환 성공 :\n",result.stdout)
        pass
    else:
        print("변환 실패 :\n",result.stderr)


def load_nii_volume(nii_path:Path):
    """
    nii 또는 .nii.gz파일의 volume을 로드

    INTPUT:
        nii_path    : .nii또는 .nii.gz파일 경로
    OUTPUT:
        volume      : 3차원 영상?
    """
    gc.collect()
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    
    file_name = nii_path.name
    ID = file_name.split("_")[0]

    

    return data, ID

def view_nii(nii_path:Path):
    """
    한개의 nii파일을 보는 함수
    INTPUT:
        nii_path            : .nii또는 .nii.gz파일 경로


    """

    img = nib.load(nii_path)
    data = img.get_fdata()

    # 초기 슬라이스 인덱스
    x, y, z = data.shape
    idxs = [x // 2, y // 2, z // 2]

    # Figure & Subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.25)

    im0 = axes[0].imshow(data[idxs[0], :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("Sagittal (X)")
    im1 = axes[1].imshow(data[:, idxs[1], :].T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal (Y)")
    im2 = axes[2].imshow(data[:, :, idxs[2]].T, cmap="gray", origin="lower")
    axes[2].set_title("Axial (Z)")

    for ax in axes:
        ax.axis("off")

    # 슬라이더 위치 정의
    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor=axcolor)
    ax_z = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)

    slider_x = Slider(ax_x, 'X-axis', 0, x - 1e-3, valinit=idxs[0], valstep=1)
    slider_y = Slider(ax_y, 'Y-axis', 0, y - 1e-3, valinit=idxs[1], valstep=1)
    slider_z = Slider(ax_z, 'Z-axis', 0, z - 1e-3, valinit=idxs[2], valstep=1)

    # 슬라이더 업데이트 함수
    def update(val):
        ix = int(slider_x.val)
        iy = int(slider_y.val)
        iz = int(slider_z.val)

        im0.set_data(data[ix, :, :].T)
        im1.set_data(data[:, iy, :].T)
        im2.set_data(data[:, :, iz].T)
        fig.canvas.draw_idle()

    # 슬라이더 이벤트 연결
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)

    plt.show()

def view_volume(volume:np.ndarray):
    """
    한개의 volume을 보는 함수
    INTPUT:
        volume            : nii로 부터 가져온 volume


    """

    
    data = volume

    # 초기 슬라이스 인덱스
    x, y, z = data.shape
    idxs = [x // 2, y // 2, z // 2]

    # Figure & Subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.25)

    im0 = axes[0].imshow(data[idxs[0], :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("Sagittal (X)")
    im1 = axes[1].imshow(data[:, idxs[1], :].T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal (Y)")
    im2 = axes[2].imshow(data[:, :, idxs[2]].T, cmap="gray", origin="lower")
    axes[2].set_title("Axial (Z)")

    for ax in axes:
        ax.axis("off")

    # 슬라이더 위치 정의
    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor=axcolor)
    ax_z = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)

    slider_x = Slider(ax_x, 'X-axis', 0, x - 1e-3, valinit=idxs[0], valstep=1)
    slider_y = Slider(ax_y, 'Y-axis', 0, y - 1e-3, valinit=idxs[1], valstep=1)
    slider_z = Slider(ax_z, 'Z-axis', 0, z - 1e-3, valinit=idxs[2], valstep=1)

    # 슬라이더 업데이트 함수
    def update(val):
        ix = int(slider_x.val)
        iy = int(slider_y.val)
        iz = int(slider_z.val)

        im0.set_data(data[ix, :, :].T)
        im1.set_data(data[:, iy, :].T)
        im2.set_data(data[:, :, iz].T)
        fig.canvas.draw_idle()

    # 슬라이더 이벤트 연결
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)

    plt.show()



def load_dcm_to_nii(input_dataset_path):
    """
    기본적인 원본 mri를 가져오는 함수

    dcm을 로드해서 nii로 저장하는 과정을 진행
    
    """
    
    # INPUT_DATASET
    #   test (1)                                다운로드 받은 콜랙션 이름
    #       ADNI                                ADNI
    #           002_S_0295                      Subject --> 환자 식별 변호
    #               MPRAGE                      촬영 방식
    #                   2012-05-10_15_44_50.0   촬영 날짜 (여러개 존재 가능)
    #                       I303066             Image Data ID --> 촬영 식별번호

    _path = input_dataset_path

    collections = [f.name for f in input_dataset_path.iterdir() if f.is_dir()]
    
    mri_count = 0

    convert_targets = []

    for collection in collections:
        _collectionPath = _path
        _path = _path / collection / 'ADNI'
        subjects = [f.name for f in _path.iterdir() if f.is_dir()]
        
        for subject in subjects:
            _subjectPath = _path
            _path = _path / subject / 'MPRAGE'
            acqDates = [f.name for f in _path.iterdir() if f.is_dir()]
            
            for acqDate in acqDates:
                _acqDatePath = _path
                _path = _path / acqDate
                imageDataIDs = [f.name for f in _path.iterdir() if f.is_dir()]
                
                for imageDataID in imageDataIDs:
                    _imageDataIDPath = _path
                    dcm_path = _path / imageDataID
                    convert_targets.append(dcm_path)
                    _path = _imageDataIDPath

                _path = _acqDatePath
            _path = _subjectPath
        _path = _collectionPath

    for idx, dcm_path in enumerate(tqdm(convert_targets, desc="dcm → nii 변환")):
        convert_dcm_to_nii(dcm_path, input_dataset_path)

 
    
    print(f"확인된 mri 개수 : {mri_count}")


def load_labels(input_dataset_path):
    """
    데이터의 정답을 가져오는 함수
    
    INPUT
        사용할 라벨 이름(vector)
    OUTPUT
        ex) 나이(vector), 성별(vector)...
    """
    
    # INPUT_DATASET 폴더속 csv확장자 모두 로드
    csv_files = list(input_dataset_path.glob("*.csv"))
    # 그중 첫번째 csv파일을 사용
    df = pd.read_csv(csv_files[0])
    df = df[['Subject','Group','Image Data ID']]
    labels = [Label(subject, group, imageID)
              for subject, group, imageID in zip(df['Subject'],df['Group'],df['Image Data ID'])]
    
    return labels

def resize_volume(volume, target_shape=(128, 128, 128)):
    """
    3D MRI 볼륨을 target_shape로 리사이즈함

    Args:
        volume (np.ndarray): 원본 MRI 볼륨 (D, H, W)
        target_shape (tuple): 원하는 출력 크기 (D, H, W)

    Returns:
        np.ndarray: 리사이즈된 MRI 볼륨
    """
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, zoom=zoom_factors, order=1)  # 선형 보간
    return resized.astype(np.float32)

def load_resample_volume(nii_path:Path, target_spacing=(1.0, 1.0, 1.0)):
    """
    NIfTI 파일을 로드하고, 지정된 voxel spacing으로 리샘플링한 3D 볼륨과 ID 반환

    INPUT:
        nii_path (Path): 입력 .nii 또는 .nii.gz 파일 경로
        target_spacing (tuple): 원하는 spacing (단위: mm)

    OUTPUT:
        np.ndarray: 리샘플링된 3D 볼륨 (float32)
        str: 파일명 기반 ID
    """
    gc.collect()
    img = nib.load(nii_path)    # 이미지 로드
    file_name = nii_path.name   
    ID = file_name.split("_")[0]    # 파일 이름에서 ID추출

    original_spacing = np.abs(img.affine[:3,:3].diagonal())
    zoom_factors = original_spacing/np.array(target_spacing)
    data = img.get_fdata()
    resampled_data = zoom(data, zoom=zoom_factors, order=1)

    # new_affine = np.copy(img.affine)
    # new_affine[:3,:3] = np.diag(target_spacing)



    return resampled_data.astype(np.float32), ID



def loader(dcm_to_nii_process:bool, size:int):
    """
    모든 기본 데이터를 갖춘 벡터를 리턴합니다.

    INPUT:
        dcm_to_nii_process : dcm파일을 nii파일로 변환을 진행여부
    
    OUTPUT:
        ClinicalDataset 리스트 혹은 벡터?
    """
    timer("초기 데이터 로드 시작")
    

    # 루트 경로 지정
    ROOT_DIR = Path(__file__).resolve().parent.parent

    # INPUT_DATASET 폴더로 주소 이동
    input_dataset_path = ROOT_DIR/"INPUT_DATASET"
    labels = load_labels(input_dataset_path)
    

    # dcm 파일을 nii로 변환
    if(dcm_to_nii_process==True):
        timer("dcm파일 nii로 변환 시작")
        load_dcm_to_nii(input_dataset_path)
        timer("dcm파일 nii로 변환 완료")

    # INPUT_DATASET에 있는 모든 .nii.gz의 이름 저장
    nii_list = sorted(input_dataset_path.glob("*.nii.gz"))
    
    # view_nii(nii_list[0])

    # 가져온 mri영상의 3차원 배열 목록
    # volumes, IDs = zip(*[load_nii_volume(i) for i in tqdm(nii_list, desc="NIfTI 로딩 중")])
    


    # 메모리 매핑
    num_sample = len(nii_list)
    example_volume, _ = load_nii_volume(nii_list[0])
    example_volume = resize_volume(example_volume, target_shape=(size, size, size))
    example_volume = np.expand_dims(example_volume, axis=-1)
        
    
    shape = example_volume.shape

    data_path = "volumes_memmap.dat"
    volumes = np.memmap(data_path, dtype=np.float32, mode='w+', shape=(num_sample, *shape))

    volume_paths = []

    save_dir = Path("volumes")
    save_dir.mkdir(parents=True, exist_ok=True)


    # 데이터 로드
    IDs =[]
    for idx, path in enumerate(tqdm(nii_list,desc="NIFTI 로딩 중")):
        volume, ID = load_nii_volume(path)
        
        volume = resize_volume(volume, target_shape=(size, size, size))
        volume = np.expand_dims(volume, axis=-1)
        
        file_path = Path("volumes")/f"{ID}.npy"
        np.save(file_path, volume)
        volume_paths.append(file_path)


        volumes[idx] = volume
        IDs.append(ID)
    
    volumes.flush()

    # 메모리 매핑 쓰기 모드 종료

    # 메모리 매핑 읽기 모드로 실행
    # volumes = np.memmap(data_path, dtype=np.float32, mode='r', shape=(num_sample, *shape))


    

    clinicalDataset = []

    # imageDataID를 기준으로 labels 딕셔너리 생성
    dict_labels = {label.imageDataID: label for label in labels}

    # IDs 순서대로 정렬된 labels 생성
    sorted_labels = [dict_labels[i] for i in IDs]
    
    # 모든 mri 영상 및 라벨 로드


    
    for i in tqdm(range(len(IDs)), desc="MRI 및 라벨 로딩 중"):
        volume_path = volume_paths[i]
        p = ClinicalDataset(volume_path,sorted_labels[i])
        clinicalDataset.append(p)

    

    timer("초기 데이터 로드 완료")
    return clinicalDataset
    

#multy processing
#기본 로드 함수
def load_and_process_nii_mp(nii_path_str: str, size: int, save_dir_str: str):
    nii_path = Path(nii_path_str)
    save_dir = Path(save_dir_str)

    # volume, ID = load_nii_volume(nii_path)
    volume, ID = load_resample_volume(nii_path)
    volume = resize_volume(volume, target_shape=(size, size, size))
    volume = np.expand_dims(volume, axis=-1)
    volume = volume.astype(np.float32) # 볼륨크기 조정-wmc

    file_path = save_dir / f"{ID}.npy"
    np.save(file_path, volume)

    return volume.astype(np.float32), ID, Path(str(file_path)) # 쒸발 이거 str아니에요 Path형 자료에요 -wmc

def load_and_process_wrapper(args):
        nii_path_str, size, save_dir_str = args
        return load_and_process_nii_mp(nii_path_str, size, save_dir_str)

def loader_parallel_process(dcm_to_nii_process: bool, size: int, max_workers: int = 8):
    """
    NIfTI 파일 로드, 전처리, 저장 과정을 멀티프로세싱으로 병렬화한 로더 함수입니다.

    INPUT
    dcm_to_nii_process: DICOM 파일을 NIfTI(.nii.gz)로 진행여부
    size: 볼륨 리사이즈 시의 목표 크기 (size x size x size).
    max_workers: 사용할 병렬 처리 프로세스 수. 기본값은 CPU 코어 수. 그냥 넣지 말고 기본값 쓰셈

    OUTPUT
    ClinicalDataset 인스턴스들의 리스트.
    """

    # 루트 경로 지정

    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    # INPUT_DATASET 폴더로 주소 이동
    input_dataset_path = ROOT_DIR / "INPUT_DATASET"
    
    labels = load_labels(input_dataset_path)

    # dcm 파일을 nii로 변환
    if dcm_to_nii_process:
        timer("dcm파일 nii로 변환 시작")
        load_dcm_to_nii(input_dataset_path)
        timer("dcm파일 nii로 변환 완료")

    # INPUT_DATASET에 있는 모든 .nii.gz의 이름 저장
    nii_list = sorted(input_dataset_path.glob("*.nii.gz"))
    
   

    # 메모리 매핑
    num_sample = len(nii_list)
    example_volume, _ = load_nii_volume(nii_list[0])
    example_volume = resize_volume(example_volume, target_shape=(size, size, size))
    example_volume = np.expand_dims(example_volume, axis=-1)
    example_volume = example_volume.astype(np.float32) # 64->32로 수정 - wmc
    shape = example_volume.shape

    data_path = "volumes_memmap.dat"
    volumes = np.memmap(data_path, dtype=np.float32, mode='w+', shape=(num_sample, *shape))

    save_dir = Path("volumes")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    #이건 병렬 프로세싱에 쓸 함수 만들기
    # func = partial(load_and_process_nii_mp, size=size, save_dir_str=str(save_dir))
    
    
    nii_path_strs = [str(p) for p in nii_list]

    clinicalDataset = []
    volume_paths = []
    IDs = []
    #여기서 병렬 프로세싱
    args_list = [(p, size, str(save_dir)) for p in nii_path_strs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(load_and_process_wrapper, args_list),
            total=num_sample,
            desc="NIFTI 멀티프로세싱 처리 중"
        ))

    for idx, (volume, ID, file_path) in enumerate(results):
        volumes[idx] = volume
        IDs.append(ID)
        volume_paths.append(file_path)

    volumes.flush()
    volumes = np.memmap(data_path, dtype=np.float32, mode='r', shape=(num_sample, *shape))

    dict_labels = {label.imageDataID: label for label in labels}
    sorted_labels = [dict_labels[i] for i in IDs]

    for i in tqdm(range(num_sample), desc="MRI 및 라벨 로딩 중"):
        p = ClinicalDataset(volume_paths[i], sorted_labels[i])
        clinicalDataset.append(p)

    timer("초기 데이터 로드 완료")
    return clinicalDataset

