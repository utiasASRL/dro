import numpy as np
import os
import os.path as osp
import sys
from itertools import accumulate
from time import time

from pyboreas.utils.odometry import (
    get_sequence_poses,
    get_sequence_poses_gt,
    get_sequences,
    calc_sequence_errors,
    get_stats,
    get_stats_per_frame,
    plot_stats,
)

default_result_path = './output'
gt_path = '/media/clegentil/ext_nvme/data/boreas_2'
dim = 2

# New sequences
sequence_type = {
        "boreas-2024-12-03-12-54" : "Suburbs",
        "boreas-2024-12-03-13-13" : "Regional",
        "boreas-2024-12-03-13-34" : "Regional",
        "boreas-2024-12-04-11-45" : "Skyway",
        "boreas-2024-12-04-11-56" : "Skyway",
        "boreas-2024-12-04-12-08" : "Skyway",
        "boreas-2024-12-04-12-19" : "Skyway",
        "boreas-2024-12-04-12-34" : "Skyway",
        "boreas-2024-12-04-14-28" : "Tunnel",
        "boreas-2024-12-04-14-34" : "Tunnel",
        "boreas-2024-12-04-14-38" : "Tunnel",
        "boreas-2024-12-04-14-44" : "Tunnel",
        "boreas-2024-12-04-14-50" : "Tunnel",
        "boreas-2024-12-04-14-59" : "Tunnel",
        "boreas-2024-12-04-15-04" : "Tunnel",
        "boreas-2024-12-04-15-10" : "Tunnel",
        "boreas-2024-12-04-15-19" : "Tunnel",
        "boreas-2024-12-04-15-24" : "Tunnel",
        "boreas-2024-12-05-14-12" : "Industrial",
        "boreas-2024-12-05-14-25" : "Suburbs",
        "boreas-2024-12-10-12-07" : "Regional",
        "boreas-2024-12-10-12-24" : "Regional",
        "boreas-2024-12-10-12-38" : "Regional",
        "boreas-2024-12-10-12-56" : "Regional",
        "boreas-2024-12-23-16-27" : "Industrial",
        "boreas-2024-12-23-16-44" : "Industrial",
        "boreas-2024-12-23-17-01" : "Industrial",
        "boreas-2024-12-23-17-18" : "Industrial",
        "boreas-2025-01-08-10-59" : "Suburbs",
        "boreas-2025-01-08-11-22" : "Suburbs",
        "boreas-2025-01-08-12-28" : "Suburbs",
        "boreas-2025-02-15-16-58" : "Suburbs",
        "boreas-2025-02-15-17-19" : "Suburbs",
        "boreas-2025-02-21-14-51" : "Suburbs",
        "boreas-2025-02-22-11-32" : "Suburbs",
        "boreas-2025-02-22-12-26" : "Suburbs",
        "boreas-2025-02-15-15-58" : "UTIAS",
        "boreas-2025-02-15-16-08" : "UTIAS",
        "boreas-2025-02-15-16-16" : "UTIAS",
        "boreas-2025-02-15-16-25" : "UTIAS",
        "boreas-2025-02-15-16-33" : "UTIAS",
        "boreas-2025-02-22-11-24" : "UTIAS",
        "boreas-2025-02-22-11-52" : "UTIAS",
        "boreas-2025-02-22-12-01" : "UTIAS",
        "boreas-2025-02-22-12-09" : "UTIAS",
        "boreas-2025-02-22-12-18" : "UTIAS",
        "boreas-2025-07-18-10-00" : "Forest",
        "boreas-2025-07-18-10-33" : "Forest",
        "boreas-2025-07-18-11-00" : "Forest",
        "boreas-2025-07-18-11-25" : "Forest",
        "boreas-2025-07-18-11-53" : "Forest",
        "boreas-2025-07-18-14-55" : "Farm",
        "boreas-2025-07-18-15-12" : "Farm",
        "boreas-2025-07-18-15-30" : "Farm",
        "boreas-2025-07-18-15-48" : "Farm",
        "boreas-2025-07-18-16-05" : "Farm",
        "boreas-2025-07-18-16-24" : "Freeway",
        "boreas-2025-08-06-06-33" : "Urban",
        "boreas-2025-08-06-07-05" : "Urban",
        "boreas-2025-08-06-07-41" : "Urban",
        "boreas-2025-08-06-08-35" : "Urban",
        "boreas-2025-08-06-10-48" : "Urban",
        "boreas-2025-08-06-11-32" : "Urban",
        "boreas-2025-08-06-12-20" : "Urban",
        "boreas-2025-08-13-07-54" : "Freeway",
        "boreas-2025-08-13-09-01" : "Farm",
        "boreas-2025-08-13-09-21" : "Farm",
        "boreas-2025-08-13-09-46" : "Farm",
        "boreas-2025-08-13-10-12" : "Farm",
        "boreas-2025-08-13-10-36" : "Farm",
        "boreas-2025-08-13-11-52" : "Freeway",
        '' : 'Unknown'
}


def main(result_path=default_result_path):
    # Get the list of folders in the result path
    folders = [f for f in os.listdir(result_path) if osp.isdir(osp.join(result_path, f))]
    folders = sorted(folders)

    # Store the results per sequence type
    results = {key: {} for key in sequence_type.values()}


    for folder in folders:
        print('Processing folder: ', folder)

        try:
            t_err, r_err, t_err_2d, r_err_2d = eval_odom(osp.join(result_path, folder, 'odometry_result'), gt_path, dim)
        except:
            print('Error in sequence: ', folder)
            continue

        print('Mean translation error: ', t_err)
        print('Mean rotation error: ', r_err)

        # Get the sequence type
        if folder not in sequence_type:
            seq_type = 'Unknown'
        else:
            seq_type = sequence_type[folder]

        results[seq_type][folder] = (t_err, r_err, t_err_2d, r_err_2d)

    print('')
    print('')
    print('Results for ', result_path, ':')

    # Print the results
    for key in results:
        print('')
        print('--------Sequence type: ', key)
        t_errs = [t_err for t_err, _, _, _ in results[key].values()]
        r_errs = [r_err for _, r_err, _, _ in results[key].values()]
        t_errs_2d = [t_err_2d for _, _, t_err_2d, _ in results[key].values()]
        r_errs_2d = [r_err_2d for _, _, _, r_err_2d in results[key].values()]
        print('Mean translation error (3D): ', np.mean(t_errs))
        print('Mean rotation error (3D): ', np.mean(r_errs))
        print('Mean translation error (2D): ', np.mean(t_errs_2d))
        print('Mean rotation error (2D): ', np.mean(r_errs_2d))
        print('Details:')
        for folder in results[key]:
            print('Sequence: ', folder, ' t_err: ', np.round(results[key][folder][0],2), '% r_err: ', np.round(results[key][folder][1],5), 'deg/m')




def compute_kitti_metrics(
    T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, plot_dir, dim, crop
):
    """Computes the translational (%) and rotational drift (deg/m) in the KITTI style.
        KITTI rotation and translation metrics are computed for each sequence individually and then
        averaged across the sequences. If 'interp' specifies a directory, we instead interpolate

        for poses at the groundtruth times and write them out as txt files.
    Args:
        T_gt (List[np.ndarray]): List of 4x4 SE(3) transforms (fixed reference frame 'i' to frame 'v', T_vi)
        T_pred (List[np.ndarray]): List of 4x4 SE(3) transforms (fixed reference frame 'i' to frame 'v', T_vi)
        seq_lens_gt (List[int]): List of sequence lengths corresponding to T_gt
        seq_lens_pred (List[int]): List of sequence lengths corresponding to T_pred
        seq (List[string]): List of sequence file names
        plot_dir (string): path to output directory for plots. Set to '' (empty string) to prevent plotting
        dim (int): dimension for evaluation. Set to '3' for SE(3) or '2' for SE(2)
        crop (List[Tuple]): sequences are cropped to prevent extrapolation, this list holds start and end indices
    Returns:
        t_err: Average KITTI Translation ERROR (%)
        r_err: Average KITTI Rotation Error (deg / m)
        t_err_2d: Average KITTI Translation ERROR (%) for 2D
        r_err_2d: Average KITTI Rotation Error (deg / m) for 2D
    """
    # set step size
    if dim == 3:
        step_size = 10  # every 10 frames should be 1 second
    elif dim == 2:
        step_size = 4  # every 4 frames should be 1 second
    else:
        raise ValueError(
            "Invalid dim value in compute_kitti_metrics. Use either 2 or 3."
        )

    # get start and end indices of each sequence
    indices_gt = [0]
    indices_gt.extend(list(accumulate(seq_lens_gt)))
    indices_pred = [0]
    indices_pred.extend(list(accumulate(seq_lens_pred)))

    # loop for each sequence
    err_list = []
    for i in range(len(seq_lens_pred)):
        ts = time()  # start time

        # get poses and times of current sequence
        T_gt_seq = T_gt[indices_gt[i] : indices_gt[i + 1]]
        T_pred_seq = T_pred[indices_pred[i] : indices_pred[i + 1]]
        # times_gt_seq = times_gt[indices_gt[i]:indices_gt[i+1]]
        # times_pred_seq = times_pred[indices_pred[i]:indices_pred[i+1]]


        if len(T_pred_seq) != len(T_gt_seq):
            T_pred_seq = T_pred_seq[crop[i][0] : crop[i][1]]

        # 2d
        err, path_lengths = calc_sequence_errors(T_gt_seq, T_pred_seq, step_size, 2)
        t_err_2d, r_err_2d, _, _ = get_stats(err, path_lengths)

        err_2d_per_frame, err_stats_2d = get_stats_per_frame(err, path_lengths)

        # 3d
        err, path_lengths = calc_sequence_errors(T_gt_seq, T_pred_seq, step_size)
        t_err, r_err, t_err_len, r_err_len = get_stats(err, path_lengths)

        err_3d_per_frame, err_stats_3d = get_stats_per_frame(err, path_lengths)

        print(seq[i], "took", str(time() - ts), " seconds")
        # print('Error: ', t_err, ' %, ', r_err, ' deg/m \n')
        print(
            f"Terr(2D) {t_err_2d:.2f}%  Rerr(2D) {r_err_2d:.4f}deg/m  Terr(3D) {t_err:.2f}% Rerr(3D) {r_err:.4f}deg/m \\\\"
        )

        err_list.append([t_err, r_err, t_err_2d, r_err_2d])

        if plot_dir:
            plot_stats(
                seq[i],
                plot_dir,
                T_pred_seq,
                T_gt_seq,
                path_lengths,
                t_err_len,
                r_err_len,
                t_err,
                r_err,
                err_2d_per_frame,
                err_stats_2d
            )

    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]
    t_err_2d = avg[2]
    r_err_2d = avg[3]

    return t_err, r_err, err_list, t_err_2d, r_err_2d




# Copy paste from pyboreas
def eval_odom(pred="test/demo/pred/3d", gt="test/demo/gt", dim=2):
    # evaluation mode

    # parse sequences
    seq = get_sequences(pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)

    # get corresponding groundtruth poses
    T_gt, _, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq, dim)

    # compute errors
    t_err, r_err, _, t_err_2d, r_err_2d = compute_kitti_metrics(
        T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred, dim, crop
    )

    # print out results
    print("Evaluated sequences: ", seq)
    print("Overall error (3D): ", t_err, " %, ", r_err, " deg/m")
    print("Overall error (2D): ", t_err_2d, " %, ", r_err_2d, " deg/m")

    return t_err, r_err, t_err_2d, r_err_2d





if __name__ == "__main__":
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        result_path = default_result_path
    main(result_path)
