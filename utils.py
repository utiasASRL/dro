import numpy as np
import cv2



def dopplerUpDown(radar_frame):
    # Remove the last folder from the radar_frame.sensor_root
    base_path = '/'.join(radar_frame.sensor_root.split('/')[:-1])
    doppler_path = base_path + '/radar/' + radar_frame.frame + '.png'
    doppler_img = cv2.imread(doppler_path, cv2.IMREAD_GRAYSCALE)
    up_chrips = doppler_img[:,10]
    return up_chrips

def checkChirp(radar_frame):
    up_chirps = dopplerUpDown(radar_frame)
    # Check that all the even chirps are up and all the odd chirps are down
    if not np.all(up_chirps[::2] == 255) or not np.all(up_chirps[1::2] == 0):
        chirp_up = True
    else:
        chirp_up = False
    return chirp_up


BOREAS_ENCODER_BIN_SIZE = 5600

def fixRadarEncoderBinSize(radar_frame, encoder_bin_size, boreas_encoder_bin_size=BOREAS_ENCODER_BIN_SIZE):
    # pyboreas hard-codes the Navtech encoder bin size (ticks per revolution)
    # used to convert the raw encoder reading into an azimuth angle to the
    # Boreas value of 5600. This function lets us apply our own encoder bin size for other data.
    if encoder_bin_size == boreas_encoder_bin_size:
        return
    encoder_ticks = np.round(radar_frame.azimuths / (2 * np.pi) * boreas_encoder_bin_size)
    radar_frame.azimuths = (encoder_ticks / encoder_bin_size * 2 * np.pi).astype(radar_frame.azimuths.dtype)
