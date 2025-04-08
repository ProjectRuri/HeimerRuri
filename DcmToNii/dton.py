import subprocess
import os

dicom_folder = 'DcmToNii/target_dcm'
output_folder = 'DcmToNii/output_nii'
os.makedirs(output_folder, exist_ok=True)

# 실행 파일 경로
dcm2niix_path = 'DcmToNii/dcm2niix.exe'

# 변환 실행
subprocess.run([
    dcm2niix_path,
    '-z', 'y',
    '-o', output_folder,
    dicom_folder
], check=True)

print("NIfTI 변환 완료!")
