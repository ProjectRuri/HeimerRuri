# HeimerRuri
프로젝트 하이머루리 (2025 캡스톤 : 치매 조기 진단)


# 뇌 부위별로 구분해주는 모델?
https://github.com/HOA-2/SlicerNeuroSegmentation?tab=readme-ov-file




# COlab dir example    
Drive    
|--template    
|--|  MNI152_T1_1mm_Brain_Mask.nii.gz - 뇌 전체 마스크    
|--|  MNI152_T1_1mm_subbr_mask.nii.gz - 소뇌 ROI 마스크    
|--|  MNI152_T1_1mm_Hipp_mask.nii.gz - 해마 ROI 마스크    
|--|  MNI152_T1_1mm_brain.nii.gz - 뇌 정합 기준 모델    
|--hd_bet    
|--|  ~~~       
|--|  HD-BET으로 두개골과 같은 불필요한 영역이 제외된 MRI영상    
|--|  ~~~    
|--model_source    
|--|  ~~~    
|--|  registration.ipynb으로 생성된 (정합orROI) 처리된 MRI 영상    
|--|  ~~~    
|--volumes_torchio_fixed    
|--|  ~~~    
|--|  반복 작업시 전처리 과정 스킵을 위한 캐시 저장폴더    
|--|  ~~~    
     
# MRI 데이터 접근 방법    
MRI 영상 데이터는 ADNI에서 제공한 ADNI2 T1을 사용    
ADNI(IDA) : https://ida.loni.usc.edu/login.jsp    

## ADNI 접근 권한 요청이 필요      
ADNI 접근 권한 요청 절차 : https://adni.loni.usc.edu/data-samples/adni-data/      

계정 생성이후 위 링크에서 데이터 이용 계약 동의 및 신청 서류 작성 필요   
작성이후 1주일 이내 이메일로 답변받는 방식으로 진행

## ADNI 2 + T1 데이터 접근   
위 방법으로 접근 권한이 성공적으로 받았을 경우 위 프로젝트에서 사용한 데이터에 접근 하기 위해선 "고급 이미지 검색"(Advanced Image Search)을 진행이 필요    

![adni_image](./IMAGE/image.png)


허가 받은 계정으로 IDA로그인 진행시 위에서 접근 가능    

사용한 이미지는 "Advanced Image Search"에서 기본 옵션중 수정한 것은 아래와 같다.   
PROJECT/PHASE : ADNI 2    
STUDY/VISIT : ADNI 2 가 포함된 모든 옵션    
IMAGE/MODALITY : MRI   
IMAGE/IMAGING PROTOCOL/MRI/WEIGHTING : T1

필요에 따라 수정가능하다.    

위 과정을 거져 검색을 진행할 경우 CSV파일과 같이 환자+촬영 목록이 출력   
우측 상단에서 SELECT ALL->ADD TO COLLECTION으로 개인 COLLECTION으로 저장   

DATA COLLECTION으로 이동후 저장 한 COLLECTION을 확인     
저장할 영상을 선택 후 1-CLICK OR ADVANCED 선택해서 다운로드   

소량 -> 1-CLICK   
대량 -> ADVANCED    

다운로드 하는 부분에서 영상들에 대한 정보가 담긴 CSV 저장 가능   

ADNI에서 제공된 파일은 기본적으로 DCM파일 -> NII.GZ확장자로 변환해야 사용에 편리함 HeimerRuri.ipynb 스크립트의 load_dcm_to_nii 함수를 이용해서 변환    

     INPUT_DATASET    
       test                                    다운로드 받은 콜랙션 이름    
           ADNI                                ADNI    
               002_S_0295                      Subject --> 환자 식별 변호    
                   MPRAGE                      촬영 방식    
                       2012-05-10_15_44_50.0   촬영 날짜 (여러개 존재 가능)    
                           I303066             Image Data ID --> 촬영 식별번호    


# HeimerRuri 동작이전 필요한 사전 작업    
1. 위에서 서술한 DCM -> NII.GZ 파일 변환   
2. hd_bet.ipynb에 있는 코드 절차대로 실행해서 두개골 및 불필요한 부분 제거 작업   
3. registration.ipynb 으로 절차대로 실행(mri 정합 처리)   
3-1. registration_roi.ipynb을 이용할 경우 template 폴더에 있는 마스크를 이용해서 특정 부위를 추출 가능   
3-2. 다른 마스크가 필요할 경우 다음 링크를 참조 : https://git.fmrib.ox.ac.uk/fsl/data_standard   
4. 필요한 전처리가 모두 처리되었을 경우 model_source/(전처리 방식)으로 정리 or HeimerRuri.ipynb 으로 지정해서 수정가능

# HeimerRuri.ipynb, Setting에서 간단히 조절 가능한 옵션들     

    MRI_SIZE = 128 -> 사용한 MRI 크기 hd-bet, registration에서 사용한 크기과 동일한 사이즈를 권장    

    # 테스트할때 사용할 데이터 수 ( 학습, 증강에서 제외됨 )   
    TEST_DATA_SIZE = 50 -> train/test 에 사용할 데이터중 test를 ad, cn 각각 TEST_DATA_SIZE 만큼 TEST로 사용   

    # 사용할 데이터 경로   
    INPUT_DATASET_PATH="/content/drive/MyDrive/model_source/registered_brains_128/" -> 위 전처리된 파일 모인 폴더 지정   

    # clinical CSV 경로 + 사용할 feature 이름
    CLINICAL_CSV_PATH = "/content/drive/MyDrive/originData2_5_16_2025.csv"  # 실제 파일명에 맞게 수정
    CLINICAL_FEATURES = ["MMSE", "Age", "Gender"]  # CSV 컬럼명에 맞게 수정 (Age/AGE, Gender/PTGENDER 등)

    VERBOSE = 0 # or 1 -> TENSORFLOW 학습과정중 로그 조절(0 -> 최소화)    
    # 캐시 전역 설정 (통일형)    
    CACHE_ROOT = Path("./cache_preprocessed")    
    CACHE_VERSION = "pre_v5"      # 버전 문자열 한 곳에서 관리 -> 전처리 방식이 수정되었을 경우 캐시가 남아있다면 수정이 필요    
    CACHE_DIR = CACHE_ROOT / CACHE_VERSION    
    CFG_VERSION = CACHE_VERSION   # TorchIO 파이프라인에서 참조    