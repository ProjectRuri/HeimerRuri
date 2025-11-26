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
![alt text](image.png)