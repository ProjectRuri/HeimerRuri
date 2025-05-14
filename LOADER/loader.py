import subprocess
import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt


from typing import Optional, Dict, Any
from matplotlib.widgets import Slider
from pathlib import Path
from classModels import *
from tqdm import tqdm
from util import *


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
        print("변환 성공 :\n",result.stdout)
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

    for collection in collections:
        # 콜랙션 위치 저장
        _collectionPath = _path

        # 콜랙션 안으로 이동
        _path = _path/collection/'ADNI'
        # subject 목록 로드
        subjects = [f.name for f in _path.iterdir() if f.is_dir()]
        for subject in subjects:
            # subject 위치 저장
            _subjectPath = _path

            # subject속 MPRAGE 안까지 이동
            _path = _path/subject/'MPRAGE'

            # 촬영 날짜 로드
            acqDates = [f.name for f in _path.iterdir() if f.is_dir()]
            for acqDate in acqDates:
                #acqDate 위치 저장
                _acqDatePath = _path

                # acq 안으로 이동
                _path = _path/acqDate

                # 촬영 식별번호 로드
                imageDataIDs = [f.name for f in _path.iterdir() if f.is_dir()]
                
                for imageDataID in imageDataIDs:
                    # 이미지데이터 위치 저장
                    _imageDataIDPath = _path

                    # dcm 파일이 있는 위치로 이동
                    _path = _path/imageDataID
                    convert_dcm_to_nii(_path,input_dataset_path)

                    # 이미지데이터 위치로 복귀
                    _path = _imageDataIDPath


                mri_count +=1

                #acqDate 위치로 복귀
                _path = _acqDatePath

            # subject 위치로 복귀
            _path = _subjectPath

        # 기존 콜랙션 위치로 복귀
        _path = _collectionPath
    
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



def loader(dcm_to_nii_process:bool):
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
    # load_dcm_to_nii(input_dataset_path)

    # dcm 파일을 nii로 변환
    if(dcm_to_nii_process==True):
        timer("dcm파일 nii로 변환 시작")
        load_dcm_to_nii(input_dataset_path)
        timer("dcm파일 nii로 변환 완료")

    # INPUT_DATASET에 있는 모든 .nii.gz의 이름 저장
    nii_list = sorted(input_dataset_path.glob("*.nii.gz"))
    

    # 가져온 mri영상의 3차원 배열 목록
    volumes, IDs = zip(*[load_nii_volume(i) for i in tqdm(nii_list, desc="NIfTI 로딩 중")])
    

    clinicalDataset = []

    # imageDataID를 기준으로 labels 딕셔너리 생성
    dict_labels = {label.imageDataID: label for label in labels}

    # IDs 순서대로 정렬된 labels 생성
    sorted_labels = [dict_labels[i] for i in IDs]
    
    # 모든 mri 영상 및 라벨 로드
    for i in tqdm(range(len(IDs)), desc="MRI 및 라벨 로딩 중"):
        p = ClinicalDataset(volumes[i],sorted_labels[i])
        clinicalDataset.append(p)

    

    timer("초기 데이터 로드 완료")
    return clinicalDataset
    



