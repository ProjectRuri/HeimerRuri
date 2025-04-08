import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. NIfTI 파일 로드
img = nib.load('nii_file_viewer/test2.nii')  # 파일 경로 수정
data = img.get_fdata()

# 2. 데이터 정규화 (0~255 범위로)
data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
data = data.astype(np.uint8)

# 3. 축 별 최대 슬라이스 수
x_max, y_max, z_max = data.shape

# 4. Figure 설정
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax_x, ax_y, ax_z = axes
im_x = ax_x.imshow(data[0, :, :], cmap='gray')
im_y = ax_y.imshow(data[:, 0, :], cmap='gray')
im_z = ax_z.imshow(data[:, :, 0], cmap='gray')
ax_x.set_title('X-axis Slice')
ax_y.set_title('Y-axis Slice')
ax_z.set_title('Z-axis Slice')

for ax in axes:
    ax.axis('off')

# 5. 애니메이션 업데이트 함수
def update(frame):
    i = frame % x_max
    j = frame % y_max
    k = frame % z_max
    im_x.set_data(data[i, :, :])
    im_y.set_data(data[:, j, :])
    im_z.set_data(data[:, :, k])
    ax_x.set_title(f'X-axis Slice {i}')
    ax_y.set_title(f'Y-axis Slice {j}')
    ax_z.set_title(f'Z-axis Slice {k}')
    return [im_x, im_y, im_z]

# 6. 애니메이션 실행
ani = FuncAnimation(fig, update, frames=max(x_max, y_max, z_max), interval=100, blit=False)
plt.tight_layout()
plt.show()
