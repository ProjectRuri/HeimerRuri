import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# 1. DICOM 파일이 있는 폴더 경로
folder_path = 'dcm_file_viewer/pet_file/'  # ← 여기에 폴더 경로 입력

# 2. DICOM 파일 로딩 및 정렬
dcm_files = sorted([
    pydicom.dcmread(os.path.join(folder_path, f))
    for f in os.listdir(folder_path)
    if f.endswith('.dcm')
], key=lambda x: float(x.InstanceNumber) if 'InstanceNumber' in x else 0)

# 3. 슬라이스 이미지 스택 만들기
images = np.stack([ds.pixel_array for ds in dcm_files], axis=0)

# 4. 시각화 설정
fig, ax = plt.subplots()
im = ax.imshow(images[0], cmap='hot')  # PET는 'hot' colormap 자주 사용
title = ax.set_title("")

ax.axis('off')

# 5. 애니메이션 업데이트 함수
def update(frame):
    im.set_array(images[frame])
    title.set_text(f"Slice {frame + 1}/{images.shape[0]}")
    return [im, title]

# 6. 애니메이션 실행
ani = FuncAnimation(fig, update, frames=images.shape[0], interval=100, blit=False)
plt.tight_layout()
plt.show()
