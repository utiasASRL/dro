import yaml
import gp_doppler as gpd
import pyboreas as pb
import time
import numpy as np
import pandas as pd
import os
import utils
import cv2
import matplotlib.pyplot as plt


def main():
    # Load the configuration file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Load the data with pyboreas
    if config['data']['multi_sequence']:
        db = pb.BoreasDataset(config['data']['data_path'])
        sequences = db.sequences
    else:
        path = config['data']['data_path']
        if path[-1] == '/':
            path = path[:-1]
        base_path = '/'.join(path.split('/')[:-1])
        db = pb.BoreasDataset(base_path)
        sequence_id = config['data']['data_path'].split('/')[-1]
        sequences = []
        sequences.append(db.get_seq_from_ID(sequence_id))
        

    # Check if the output path exists
    os.makedirs('output', exist_ok=True)

    # Check the visualisation, saving, and verbose options
    visualise = config['log']['display']
    save_images = config['log']['save_images']
    verbose = config['log']['verbose']

    # Get the sensor configuration
    doppler_radar = config['radar']['doppler_enabled']
    if not doppler_radar:
        chirp_up = config['radar']['chirp_up']
    use_gyro = config['estimation']['use_gyro']

    # Parameters for bias estimation
    gyro_bias_alpha = 0.01 # For the gyro bias low-pass filter update
    estimate_gyro_bias = False

    # Select the motion model according to the configuration
    pose_estimation = True
    estimate_ang_vel = False
    if use_gyro:
        motion_model = 'const_body_vel_gyro'
    elif config['estimation']['direct_cost']:
        motion_model = 'const_vel_const_w'
        estimate_ang_vel = True
    elif config['estimation']['doppler_cost']:
        motion_model = 'const_vel'
        pose_estimation = False
    else:
        print("Ambiguous configuration: no motion model selected")
        return
    opts = config.copy()
    opts['estimation']['motion_model'] = motion_model



    # Prepare for the vy bias estimation
    if doppler_radar:
        estimate_vy_bias = config['estimation']['estimate_doppler_vy_bias']
        if estimate_vy_bias:
            T_axle_radar = np.array(config['estimation']['T_axle_radar'])
            if 'vy_bias_alpha' in config['estimation']:
                vy_bias_alpha = config['estimation']['vy_bias_alpha']
            else:
                vy_bias_alpha = 0.01
            vy_bias = 0.0
    else:
        estimate_vy_bias = False
    

    if visualise and pose_estimation:
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        # Set axis equal
        ax.set_aspect('equal', adjustable='datalim')
        plt.draw()
        plt.pause(0.001)
        gt_display = None
        est_display = None
        ax.set_title('Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')



    # Loop over the sequences
    for seq in sequences:

        # Create the GP model
        temp_radar_frame = seq.get_radar(0)
        res = temp_radar_frame.resolution
        state_estimator = gpd.GPStateEstimator(opts, res)
        temp_radar_frame.unload_data()

        # For logging trajectory and display
        gt_first_T_inv= None
        gt_xyz = []
        est_xyz = []
        

        # If using the gyro, we need to load the IMU data
        if use_gyro:
            gyro_bias = 0.0
            gyro_bias_counter = 0
            gyro_bias_initialised = False
            previous_vel_null = False
            estimate_gyro_bias = config['estimation']['gyro_bias_estimation']

            # Need to account for the IMU type
            # (in original Boreas, there is no independent IMU)
            if config['imu']['type'] == 'applanix':
                imu_path = os.path.join(seq.seq_root, 'applanix', 'imu_raw.csv')
                imu_data = np.loadtxt(imu_path, delimiter=',', skiprows=1)
                imu_time = imu_data[:, 0]
                imu_gyro = np.stack((imu_data[:, 3], imu_data[:, 2], imu_data[:, 1]), axis=1)
                T_applanix_radar = seq.calib.T_applanix_lidar @ np.linalg.inv(seq.calib.T_radar_lidar)
                imu_gyro = imu_gyro @ T_applanix_radar[:3, :3]
                imu_yaw = -imu_gyro[:, 2]
            elif config['imu']['type'] == 'dmu':
                imu_path = os.path.join(seq.seq_root, 'imu', 'dmu_imu.csv')
                imu_data = np.loadtxt(imu_path, delimiter=',', skiprows=1)
                imu_time = imu_data[:, 0] * 1e-9
                imu_yaw = imu_data[:, 3]
            else:
                print("Unknown IMU type")
                return

            # Give all the IMU data to the state estimator
            state_estimator.motion_model.setGyroData(imu_time, imu_yaw)
            min_gyro_sample_bias = config['imu']['min_time_bias_init'] / np.mean(np.diff(imu_time))
            if estimate_vy_bias:
                vy_bias = config['estimation']['vy_bias_prior']


            if 'gyro_bias_prior' in config['estimation']:
                gyro_bias = config['estimation']['gyro_bias_prior']
                gyro_bias_initialised = True
                gyro_bias_counter = min_gyro_sample_bias + 1




        # Prepare output folders
        seq_output_path = 'output/' + seq.ID
        os.makedirs(seq_output_path, exist_ok=True)
        odom_output_path = seq_output_path + '/odometry_result'
        if os.path.exists(odom_output_path):
            os.system('rm -r ' + odom_output_path)
        os.makedirs(odom_output_path)
        odom_output_path = odom_output_path + '/' + seq.ID + '.txt'
        other_log_path = seq_output_path + '/other_log'
        if os.path.exists(other_log_path):
            os.system('rm -r ' + other_log_path)
        os.makedirs(other_log_path, exist_ok=True)
        if save_images:
            image_output_path = seq_output_path + '/images'
            if os.path.exists(image_output_path):
                os.system('rm -r ' + image_output_path)
            os.makedirs(image_output_path, exist_ok=True)



        # Variables to log the time
        time_sum = 0
        opti_time_sum = 0
        time_counter = 0

        # Tracking of the chirp up for doppler radar
        if not doppler_radar:
            chirp_up = not config['radar']['chirp_up']

        
        # Loop over the radar frames
        end_id = len(seq.radar_frames)
        start_id = 0
        for i in range(start_id, end_id):
            time_start = time.time()

            # Load the radar frame
            radar_frame = seq.get_radar(i)
            
            if gt_first_T_inv is None:
                gt_first_T_inv = np.linalg.inv(radar_frame.pose)
            gt_xyz.append((gt_first_T_inv @ radar_frame.pose)[:3,3])


            # Update the gyro bias if needed
            if use_gyro and estimate_gyro_bias and gyro_bias_initialised:
                state_estimator.motion_model.setGyroBias(gyro_bias)

            
            # Check the chirp up/down status to account for the
            # hardware problem of the doppler radar
            if doppler_radar:
                chirp_up = utils.checkChirp(radar_frame)
            

            # Display of the progress
            if time_counter == 0:
                print("Frame " + str(i-start_id+1) + " / " + str(end_id-start_id), end='\r')
            else:
                print("Frame " + str(i-start_id+1) + " / " + str(end_id-start_id) + " - Avg. opti: " + str(round(opti_time_sum/time_counter,3)) + "s, time left (including visualisation): " + str(round((end_id-i)*time_sum/time_counter/60, 3)) + "min    ", end='\r')



            # Optimise the scan velocity
            time_start = time.time()
            polar_img = radar_frame.polar
            # Dirty way to account for the offset
            offset = config['radar']['range_offset'] / radar_frame.resolution
            if offset > 0:
                polar_img = np.concatenate(np.zeros((polar_img.shape[0], int(np.round(offset)))), polar_img, axis=1)
            elif offset < 0:
                polar_img = polar_img[:, int(np.round(-offset)):]
                

            state = state_estimator.odometryStep(polar_img, radar_frame.azimuths.flatten(), radar_frame.timestamps.flatten(), chirp_up)

            
            # Get the velocity 
            velocity = state[:2]
            if config['estimation']['motion_model'] == 'const_body_acc_gyro':
                velocity = state[:2] * (1+state[2]*0.125)
            if estimate_vy_bias and np.linalg.norm(velocity) > 3:
                state_estimator.vy_bias = 0.0
                doppler_vel = state_estimator.getDopplerVelocity()
                doppler_vel = np.concatenate([doppler_vel, [0]])
                print("\nDoppler velocity: ", doppler_vel)
                if use_gyro:
                    # Get the average angular velocity between the first and last azimuth
                    gyro_idx = np.logical_and(imu_time >= radar_frame.timestamps[0]*1e-6, imu_time <= radar_frame.timestamps[-1]*1e-6)
                    gyro_data = imu_yaw[gyro_idx]
                    if len(gyro_data) == 0:
                        gyro_data = np.array([0, 0, 0])
                    else:
                        gyro_data = np.mean(gyro_data)
                        gyro_data = T_axle_radar[:3, :3] @ np.array([0, 0, gyro_data])
                    axle_vel = T_axle_radar[:3, :3] @ doppler_vel + np.cross(gyro_data, T_axle_radar[3, :3])
                else:
                    axle_vel = T_axle_radar[:3, :3] @ doppler_vel
                vy = (T_axle_radar[:3,:3].T@(np.array([0, axle_vel[1], 0])))[1]
                vy_bias = vy_bias_alpha * vy + (1-vy_bias_alpha) * vy_bias
                state_estimator.vy_bias = vy_bias
            if estimate_vy_bias:
                print("\nVy bias: ", vy_bias)

            time_end = time.time()


            # Time the optimisation (remove the first few warmup iterations)
            if time_counter == 5:
                opti_time_sum = (time_end - time_start)*5
            opti_time_sum += time_end - time_start


            # Display information
            if verbose:
                print("\n")
                print("Velocity: ", velocity)
                print("Velocity GT: ", radar_frame.body_rate[:2].flatten())
                print("Diff norm: ", np.linalg.norm(velocity) - np.linalg.norm(radar_frame.body_rate[:3]))
                print("\n")

            # Log the velocity in a csv file with timestamp_scan, timestamp_min, timstamp_max, x velocity, y velocity
            vel_pd = pd.DataFrame(np.concatenate([np.array(radar_frame.timestamp).reshape(-1,1), np.array(radar_frame.timestamps.min()).reshape(-1,1), np.array(radar_frame.timestamps.max()).reshape(-1,1), velocity.reshape(1, -1)], axis=1).reshape(1, -1))
            vel_pd[1] = vel_pd[1].astype(int)
            vel_pd[2] = vel_pd[2].astype(int)
            vel_pd[1] = radar_frame.timestamps.min()
            vel_pd[2] = radar_frame.timestamps.max()
            if not os.path.exists(other_log_path + '/velocity.csv'):
                vel_pd.to_csv(other_log_path + '/velocity.csv', header=['timestamp_scan (s)', 'timestamp_min (us)', 'timestamp_max (us)', 'vx', 'vy'], index=None)
            else:
                vel_pd.to_csv(other_log_path + '/velocity.csv', mode='a', header=False, index=None)



            # Estimate the gyro bias when the velocity is null
            if estimate_gyro_bias and np.linalg.norm(velocity) < 0.05:
                if previous_vel_null:
                    # Get gyro measurements between the first and last azimuth
                    gyro_idx = np.logical_and(imu_time >= radar_frame.timestamps[0]*1e-6, imu_time <= radar_frame.timestamps[-1]*1e-6)
                    gyro_data = imu_yaw[gyro_idx]
                    invalid = False
                    if gyro_bias_counter != 0 and (np.abs(np.mean(gyro_data)-gyro_bias) > 2*np.abs(gyro_bias)):
                        invalid = True
                    if not invalid:
                        if not gyro_bias_initialised:
                            gyro_bias += np.sum(gyro_data)
                            gyro_bias_counter += len(gyro_data)
                            if gyro_bias_counter > min_gyro_sample_bias:
                                gyro_bias /= gyro_bias_counter
                                gyro_bias_initialised = True
                        else:
                            gyro_bias = gyro_bias_alpha * np.mean(gyro_data) + (1-gyro_bias_alpha) * gyro_bias
                previous_vel_null = True
            if estimate_gyro_bias and verbose:
                if gyro_bias_initialised:
                    print("Gyro bias: ", gyro_bias)
                else:
                    print("Gyro bias not initialised")





            # Get the position and rotation for the evaluation
            # (account the calibrated angular velocity bias
            # when not using the gyro  mentioned in the paper)
            if estimate_ang_vel :
                state_estimator.state_init[2] -= config['estimation']['ang_vel_bias']
            current_pos, current_rot = state_estimator.getAzPosRot()
            if estimate_ang_vel:
                state_estimator.state_init[2] += config['estimation']['ang_vel_bias']

            # Get the GT position and rotation
            if current_pos is not None:
                current_pos = current_pos.squeeze()
                current_rot = current_rot.squeeze()
                
                # Get the id closest to the GT (should query 
                # the exact time instead of the closest, but 
                # that the way it is done for now)
                mid_id = np.argmin(np.abs(radar_frame.timestamps.flatten().astype(np.float64)*1e-6 - radar_frame.timestamp))

                trans_mat = np.array([[np.cos(current_rot[mid_id]), -np.sin(current_rot[mid_id]), 0, current_pos[mid_id][0]],
                                    [np.sin(current_rot[mid_id]), np.cos(current_rot[mid_id]), 0, current_pos[mid_id][1]],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

                est_xyz.append(trans_mat[:3,3])

                # Save the odometry result for the pyboreas evaluation script
                trans_mat = np.linalg.inv(trans_mat)
                data = np.concatenate([radar_frame.timestamps[mid_id], trans_mat[:3, :].flatten()]).reshape(1, -1)
                df_data = pd.DataFrame(data)
                df_data[0] = df_data[0].astype(int)
                if not os.path.exists(odom_output_path):
                    df_data.to_csv(odom_output_path, header=None, index=None, sep=' ')
                else:
                    df_data.to_csv(odom_output_path, mode='a', header=None, index=None, sep=' ')



            # Visualisation and image saving
            if visualise or save_images:
                img = state_estimator.generateVisualisation(radar_frame, 500, radar_frame.resolution*radar_frame.polar.shape[1]/(250*np.sqrt(2)), inverted=True, text=True)
            if visualise:
                cv2.imshow('Image', img)
                cv2.waitKey(5)

                if pose_estimation:
                    est_xy = np.array(est_xyz)
                    gt_xy = np.array(gt_xyz)
                    if est_display is None:
                        est_display, = ax.plot(est_xy[:,0], est_xy[:,1], 'b-', label='Estimate')
                    else:
                        est_display.set_xdata(est_xy[:,0])
                        est_display.set_ydata(est_xy[:,1])
                    if gt_display is None:
                        gt_display, = ax.plot(gt_xy[:,0], gt_xy[:,1], 'r--', label='GT')
                        ax.scatter(gt_xy[:,0], gt_xy[:,1], marker='s', color='k', s=100, label='Sequence start')
                        ax.legend()
                    else:
                        gt_display.set_xdata(gt_xy[:,0])
                        gt_display.set_ydata(gt_xy[:,1])

                    ax.set_xlim(min(np.min(gt_xy[:,0]), np.min(est_xy[:,0]))-100, max(np.max(gt_xy[:,0]), np.max(est_xy[:,0]))+100)
                    ax.set_ylim(min(np.min(gt_xy[:,1]), np.min(est_xy[:,1]))-100, max(np.max(gt_xy[:,1]), np.max(est_xy[:,1]))+100)
                    plt.draw()
                    plt.pause(0.001)
            if save_images:
                cv2.imwrite(image_output_path + '/frame_' + str(i-start_id).zfill(6) + '.png', img)




            radar_frame.unload_data()


            # Time the loop for statistics
            time_end = time.time()
            if time_counter == 1:
                time_sum = time_end - time_start
            time_sum += time_end - time_start
            time_counter += 1


    if visualise:
        cv2.destroyAllWindows()
    print("")




if __name__ == '__main__':
    main()
