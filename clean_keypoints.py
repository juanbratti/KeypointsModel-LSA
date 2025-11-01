import h5py
import numpy as np
from tqdm import tqdm  # <-- barra de progreso

# --- Índices de subsets según MediaPipe Holistic ---
POSE_IDX = np.arange(0, 33)         # 33 keypoints
FACE_IDX = np.arange(33, 501)       # 468 keypoints
LEFT_HAND_IDX = np.arange(501, 522) # 21 keypoints
RIGHT_HAND_IDX = np.arange(522, 543)# 21 keypoints

SUBSETS = {
    "pose": POSE_IDX,
    "face": FACE_IDX,
    "left_hand": LEFT_HAND_IDX,
    "right_hand": RIGHT_HAND_IDX,
}

def impute_sequence(seq):
    """
    Imputa NaNs en coordenadas X,Y usando interpolación, extrapolación y ceros.
    seq: np.array (frames, 543, 4)
    """
    seq_out = seq.copy()

    for name, idx in SUBSETS.items():
        subset = seq[:, idx, :2]  # Solo X,Y
        nan_mask = np.isnan(subset)

        if nan_mask.all():
            # Caso extremo: subset completo ausente -> zeros
            seq_out[:, idx, :2] = 0.0
            continue

        for k in range(len(idx)):  # cada keypoint dentro del subset
            arr = subset[:, k]  # (frames, 2) -> X,Y

            for dim in range(2):  # 0=x, 1=y
                vals = arr[:, dim]
                nans = np.isnan(vals)

                if nans.all():
                    vals[:] = 0.0
                else:
                    not_nan = ~nans
                    # Interpolación interna
                    vals[nans] = np.interp(
                        np.flatnonzero(nans),
                        np.flatnonzero(not_nan),
                        vals[not_nan]
                    )
                    # Extrapolación en bordes
                    first_valid = np.flatnonzero(not_nan)[0]
                    last_valid = np.flatnonzero(not_nan)[-1]
                    vals[:first_valid] = vals[first_valid]
                    vals[last_valid+1:] = vals[last_valid]

                arr[:, dim] = vals
            subset[:, k] = arr

        seq_out[:, idx, :2] = subset

    return seq_out


def clean_h5(in_path="keypoints.h5", out_path="keypoints_cleaned.h5"):
    """
    Lee el archivo H5, aplica imputación a los keypoints y guarda un nuevo archivo limpio.
    """
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        clips = list(fin.keys())
        for clip_id in tqdm(clips, desc="Procesando clips"):
            clip_group_out = fout.create_group(clip_id)
            clip_group_in = fin[clip_id]

            for signer in clip_group_in.keys():
                signer_group_out = clip_group_out.create_group(signer)
                signer_group_in = clip_group_in[signer]

                # --- Procesar keypoints ---
                kp = signer_group_in["keypoints"][:]  # (frames, 2172)
                kp = kp.reshape(kp.shape[0], 543, 4)

                kp_clean = impute_sequence(kp)

                # Guardar keypoints (aplanados otra vez)
                signer_group_out.create_dataset(
                    "keypoints", data=kp_clean.reshape(kp.shape[0], -1)
                )

                # --- Copiar boxes sin cambios ---
                boxes = signer_group_in["boxes"][:]
                signer_group_out.create_dataset("boxes", data=boxes)

    print(f"Dataset limpio guardado en: {out_path}")


if __name__ == "__main__":
    clean_h5("keypoints.h5", "keypoints_cleaned.h5")
