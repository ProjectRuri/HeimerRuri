import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


def load_dicom_slices(folder_path):
    slices = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".dcm"):
            fpath = os.path.join(folder_path, fname)
            ds = pydicom.dcmread(fpath)
            if hasattr(ds, 'ImagePositionPatient'):
                slices.append(ds)

    # Z축 정렬
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    return slices


def apply_lut_to_slice(slice):
    img = apply_modality_lut(slice.pixel_array, slice)
    img = apply_voi_lut(img, slice)
    return img


def animate_3_planes_lut(slices):
    # LUT 적용된 볼륨 생성
    volume = np.stack([apply_lut_to_slice(s) for s in slices], axis=0)

    z, y, x = volume.shape
    size = max(x, y, z)

    def square_canvas(slice_2d):
        canvas = np.zeros((size, size), dtype=slice_2d.dtype)
        h, w = slice_2d.shape
        h_offset = (size - h) // 2
        w_offset = (size - w) // 2
        canvas[h_offset:h_offset+h, w_offset:w_offset+w] = slice_2d
        return canvas

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(square_canvas(volume[0, :, :]), cmap='gray')
    im1 = ax1.imshow(square_canvas(np.rot90(volume[:, 0, :], 3)), cmap='gray')
    im2 = ax2.imshow(square_canvas(np.rot90(volume[:, :, 0], 3)), cmap='gray')

    ax0.set_title('Axial')
    ax1.set_title('Coronal')
    ax2.set_title('Sagittal')
    for ax in axes:
        ax.axis('off')

    max_frames = max(z, y, x)

    def update(frame):
        i = frame % z
        j = frame % y
        k = frame % x

        axial = square_canvas(volume[i, :, :])
        coronal = square_canvas(np.rot90(volume[:, j, :], 3))
        sagittal = square_canvas(np.rot90(volume[:, :, k], 3))

        im0.set_data(axial)
        im1.set_data(coronal)
        im2.set_data(sagittal)

        ax0.set_title(f'Axial (z={i})')
        ax1.set_title(f'Coronal (y={j})')
        ax2.set_title(f'Sagittal (x={k})')
        return [im0, im1, im2]

    ani = FuncAnimation(fig, update, frames=max_frames, interval=100, blit=False)
    plt.tight_layout()
    plt.show()


slices = load_dicom_slices('dcm_file_viewer/mri_file')
animate_3_planes_lut(slices)