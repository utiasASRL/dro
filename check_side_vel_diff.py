import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from pyboreas.utils.utils import yawPitchRollToRot


def main():
    #displayDifferences()
    displayGTErrors()


# Get the sequence path and ID from the config file
def getSequenceInfo():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    data_path = config['data']['data_path']
    multi_seq = config['data']['multi_sequence']
    data_paths = []
    if multi_seq:
        paths = os.listdir(data_path)
        for path in paths:
            if os.path.isdir(os.path.join(data_path, path)):
                if path[-1] == '/':
                    path = path[:-1]
                data_paths.append(os.path.join(data_path, path))
    else:
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        data_paths = [data_path]
    # sort the data paths for consistency
    data_paths.sort()
    return data_paths


# Read the log of the Doppler-only vs DRO velocities
def loadLog(seq_id):
    return pd.read_csv('output/' + seq_id + '/other_log/doppler_vs_dro_velocity.csv')


# Read the Applanix GT poses and express the ENU velocities in the radar frame
def loadGTBodyVelocity(seq_path):
    gt = np.loadtxt(os.path.join(seq_path, 'applanix', 'radar_poses.csv'), delimiter=',', skiprows=1)
    t_gt = gt[:, 0] * 1e-6
    vel_enu = gt[:, 4:7]
    # Columns 7, 8, 9 are roll, pitch, heading
    body_vel = np.empty_like(vel_enu)
    for i in range(gt.shape[0]):
        rot_enu_radar = yawPitchRollToRot(gt[i, 9], gt[i, 8], gt[i, 7])
        body_vel[i] = rot_enu_radar.T @ vel_enu[i]
    return t_gt, body_vel


# Match each logged scan with the closest GT sample
def matchGT(t_log, t_gt, body_vel):
    idx = np.argmin(np.abs(t_gt.reshape(1, -1) - t_log.reshape(-1, 1)), axis=1)
    return body_vel[idx]


def displayDifferences():
    seq_paths = getSequenceInfo()
    dfs = []
    for seq_path in seq_paths:
        seq_id = seq_path.split('/')[-1]
        try:
            df = loadLog(seq_id)
        except FileNotFoundError:
            continue
        dfs.append(df)
    fig, ax = plt.subplots(len(dfs)//3 + 1, 3, figsize=(15, 5 * (len(dfs)//3 + 1)))
    for i, df in enumerate(dfs):
        row = i // 3
        col = i % 3
        ax[row, col].plot(df['diff_vy'], label='diff_vy (mean={:.4f})'.format(df['diff_vy'].mean()))
        ax[row, col].legend()
        ax[row, col].grid(True)
    differences = pd.concat([df['diff_vy'] for df in dfs])
    plt.figure()
    plt.plot(differences, label='diff_vy (mean={:.4f})'.format(differences.mean()))
    plt.legend()
    plt.show(block=False)


# Plot the error of the Doppler-only and DRO lateral velocities with respect to the GT
def displayGTErrors():
    seq_paths = getSequenceInfo()
    n_rows = 3
    n_cols = len(seq_paths)//3 + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True, sharey=True)
    for i, seq_path in enumerate(seq_paths):
        seq_id = seq_path.split('/')[-1]
        try:
            df = loadLog(seq_id)
            t_log = df['timestamp_scan (s)'].values
        except (FileNotFoundError, KeyError):
            continue

        t_gt, body_vel = loadGTBodyVelocity(seq_path)
        gt_vel = matchGT(t_log, t_gt, body_vel)
        gt_vy = gt_vel[:, 1]

        doppler_err = df['doppler_vy_corrected'].values - gt_vy
        dro_err = df['dro_vy'].values - gt_vy

        time = t_log - t_log[0]

        row = i % 3
        col = i // 3

        ax[row, col].plot(time, df['doppler_vy_corrected'].values, label='Doppler-only', linewidth=1)
        ax[row, col].plot(time, df['dro_vy'].values, label='DRO', linewidth=1, alpha=0.7)
        ax[row, col].plot(time, gt_vy, label='GT', color='k', linewidth=1, alpha=0.7)

        #for err, name in ((doppler_err, 'Doppler-only'), (dro_err, 'DRO')):
        #    label = '{} (mean={:.4f})'.format(
        #        name, np.mean(err))
        #    ax[row, col].plot(time, err, label=label, linewidth=1)
        ax[row, col].set_title('Lateral velocity (sequence ' + seq_id + ')')
        ax[row, col].legend()
        ax[row, col].grid(True)

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
