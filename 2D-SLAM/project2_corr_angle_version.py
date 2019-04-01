import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.signal import butter, lfilter
import cv2


def tic():
    return time.time()


def toc(tstart, name="Operation"):
    print('%s took: %s sec.\n' % (name, (time.time() - tstart)))


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class MyVehicle:
    def __init__(self, dataset, num_ptc=100, do_texture=True, angle_shift=False, particle_shift=False, name=""):
        self.name = name
        if self.name != "":
            self.name = name + "_"
        self.do_texture = do_texture
        self.angle_shift = angle_shift
        self.dataset = dataset
        self.particle_shift = particle_shift
        self.map_update_val = 9
        self.input_noise = [[0, 0.3], [0, 0.5]]
        self.height_threshold = [-2, 0.35]
        self.num_ptc = num_ptc
        self.resample_threshold = 0.5
        self.draw_threshold = 1 * np.log(self.map_update_val)
        self.dir_name = self.name + "result_{}".format(self.dataset)
        self.camera_calibration = np.array([[585.05108211, 0, 242.94140713],
                                            [0, 585.05108211, 315.83800193],
                                            [0, 0, 1]])
        self.camera_roc = np.array([[0, -1, 0],
                                    [0, 0, -1],
                                    [1, 0, 0]])
        self.inv_camera_roc = self.camera_roc.T

        self.inv_camera_calibration = np.linalg.inv(self.camera_calibration)
        self.camera_position = np.array([0.18, 0.005, 0.36])
        self.camera_rpy = np.array([0.0, 0.36, 0.021])  # rad
        self.rotation_camera_cam2body_yaw = np.array([[np.cos(0.021), -np.sin(0.021), 0],
                                                      [np.sin(0.021), np.cos(0.021), 0],
                                                      [0, 0, 1]])

        self.rotation_camera_cam2body_pitch = np.array([[np.cos(0.36), 0, np.sin(0.36)],
                                                        [0, 1, 0],
                                                        [-np.sin(0.36), 0, np.cos(0.36)]])

        self.rotation_camera_cam2body_roll = np.array([[1, 0, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 1]])

        self.rotation_camera_cam2body = np.dot(np.dot(self.rotation_camera_cam2body_yaw,
                                                      self.rotation_camera_cam2body_pitch),
                                               self.rotation_camera_cam2body_roll)
        self.transform_cam2body = np.hstack((self.rotation_camera_cam2body, self.camera_position.reshape(-1, 1)))
        self.transform_cam2body = np.vstack((self.transform_cam2body, np.array([[0, 0, 0, 1]])))

        self.lidar_position = np.array([0.13323, 0]).reshape(2, 1)

        if not os.path.exists(self.dir_name):
            print("create path:", self.dir_name)
            os.mkdir("./" + self.dir_name)

        with np.load("Encoders%d.npz" % dataset) as data:
            self.encoder_counts = data["counts"]  # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"]  # encoder time stamps

        with np.load("Hokuyo%d.npz" % dataset) as data:
            self.lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
            self.lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
            self.lidar_range_min = data["range_min"]  # minimum range value [m]
            self.lidar_range_max = data["range_max"]  # maximum range value [m]
            self.lidar_ranges = data[
                "ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans

        with np.load("Imu%d.npz" % dataset) as data:

            self.imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
            self.imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
            order = 1
            fs = 100
            cutoff = 10
            self.imu_angular_velocity[2, :] = butter_lowpass_filter(self.imu_angular_velocity[2, :], cutoff, fs, order)
            # plt.figure()
            # plt.plot(self.imu_angular_velocity[2,:])
            # plt.show()
        if self.do_texture:
            with np.load("Kinect%d.npz" % dataset) as data:
                self.disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
                self.rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images

        self.trajectory_pre = self.pre_trajectory()

        self.lidar_angles = np.arange(self.lidar_angle_min,
                                      self.lidar_angle_max + 0.5 * self.lidar_angle_increment,
                                      self.lidar_angle_increment)

        self.MAP = self.init_map()


    def init_map(self):
        MAP = {}
        MAP['res'] = 0.05  # meters
        MAP['xmin'] = -10  # meters -10 -30 40 20
        MAP['ymin'] = -10
        MAP['xmax'] = 30
        MAP['ymax'] = 30
        MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
        MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8
        MAP['log_map'] = np.zeros((MAP['sizex'], MAP['sizey']))
        MAP['texture_map'] = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
        state = np.zeros((3, 1))

        ranges = self.lidar_ranges[:, 0]
        # take valid indices
        indValid = np.logical_and((ranges < self.lidar_range_max), (ranges > self.lidar_range_min))
        ranges = ranges[indValid]
        angles = self.lidar_angles[indValid]
        body_detect = self.lidar2body(ranges, angles)
        world_detect = self.body2world(body_detect, state)
        world_detect = np.squeeze(world_detect, 2)

        body_lidar = self.lidar2body(0, 0)
        world_lidar = self.body2world(body_lidar, state)
        world_lidar = np.squeeze(world_lidar, 2)

        xy_is = np.ceil((world_detect - np.array([MAP['xmin'], MAP['ymin']]).reshape(2, 1)) / MAP['res']).astype(np.int32) - 1
        lidar_is = np.ceil((world_lidar - np.array([MAP['xmin'], MAP['ymin']]).reshape(2, 1)) / MAP['res']).astype(np.int32) - 1
        xy_occupied = np.hstack((xy_is, lidar_is))
        xy_occupied_idx = np.vstack((xy_occupied[1, :], xy_occupied[0, :])).T
        xy_occupied_idx = np.expand_dims(xy_occupied_idx, 0)
        MAP['log_map'][xy_is[0, :], xy_is[1, :]] += 2 * np.log(self.map_update_val)  # may invert

        mask_map = np.zeros_like(MAP['map'])
        cv2.drawContours(image=mask_map, contours=[xy_occupied_idx], contourIdx=0,
                         color=np.log(1.0 / self.map_update_val), thickness=-1)

        MAP['log_map'] += mask_map

        occupied = MAP['log_map'] > 0
        empty = MAP['log_map'] <= 0
        MAP['map'][occupied] = 1
        MAP['map'][empty] = 0
        # plt.figure()
        # plt.imshow(MAP['map'])
        # plt.show()
        return MAP

    def read_disparity(self, idx_disp):
        file_name = "./dataRGBD/Disparity{}/disparity{}_{}.png".format(self.dataset, self.dataset, idx_disp + 1)
        img_test = cv2.imread(file_name, -1)
        dd = (-0.00304 * img_test + 3.31)
        depth = 1.03 / dd
        res = np.meshgrid(np.arange(0, img_test.shape[1]), np.arange(0, img_test.shape[0]))

        i = res[0]
        j = res[1]
        rgbi = (i * 526.37 + dd * (-4.5 * 1750.46) + 19276.0) / 585.051
        rgbj = (j * 526.37 + 16662.0) / 585.051
        depth = depth.reshape(-1, 1)
        rgbi = rgbi.reshape(-1, 1)
        rgbj = rgbj.reshape(-1, 1)
        # flag1 = np.logical_and(rgbi >= 0, rgbi <= img_test.shape[0])
        # flag2 = np.logical_and(rgbj >= 0, rgbj <= img_test.shape[1])

        flag1 = np.logical_and(rgbi >= 0, rgbi <= img_test.shape[1])
        flag2 = np.logical_and(rgbj >= 0, rgbj <= img_test.shape[0])
        flag3 = np.logical_and(flag1, flag2)
        indValid = np.logical_and(depth >= 0, flag3)
        rgbi = rgbi[indValid]
        rgbj = rgbj[indValid]
        depth = depth[indValid]
        return depth, np.round(rgbi).astype(np.int16), np.round(rgbj).astype(np.int16)

    def read_rgb_image(self, idx_rgbd):
        file_name = "./dataRGBD/RGB{}/rgb{}_{}.png".format(self.dataset, self.dataset, idx_rgbd + 1)
        img_test = cv2.imread(file_name)
        # img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
        return img_test


    def run_main(self):
        state = np.zeros((3, self.num_ptc))
        weight = np.ones((1, self.num_ptc), np.float64) / self.num_ptc

        idx_imu = 0
        idx_lidar = 0
        idx_rgbd = 0
        idx_disp = 0
        trajectory = np.zeros((3, 0))

        xs = np.arange(-2, 3)
        ys = np.arange(-2, 3)
        theta_s = np.arange(-3, 4) * np.pi / 360

        for idx_encoder in range(200, self.encoder_stamps.shape[0]):
            if idx_encoder == 300:
                self.draw_threshold = 3 * np.log(self.map_update_val)

            while idx_imu + 1 < self.imu_stamps.shape[0] and \
                    self.encoder_stamps[idx_encoder] >= self.imu_stamps[idx_imu + 1]:
                idx_imu = idx_imu + 1

            while idx_lidar + 1 < self.lidar_stamsp.shape[0] and \
                    self.encoder_stamps[idx_encoder] >= self.lidar_stamsp[idx_lidar + 1]:
                idx_lidar = idx_lidar + 1

            if idx_lidar + 1 < self.lidar_stamsp.shape[0] and \
                   np.abs(self.encoder_stamps[idx_encoder] - self.lidar_stamsp[idx_lidar]) >= \
                   np.abs(self.encoder_stamps[idx_encoder] - self.lidar_stamsp[idx_lidar + 1]):
                idx_lidar = idx_lidar + 1
            # print("lidar difference", np.abs(self.encoder_stamps[idx_encoder] - self.lidar_stamsp[idx_lidar]))
            # print("imu difference", np.abs(self.encoder_stamps[idx_encoder] - self.imu_stamps[idx_imu]))
            if self.do_texture:
                while idx_disp + 1 < self.disp_stamps.shape[0] and \
                        self.encoder_stamps[idx_encoder] >= self.disp_stamps[idx_disp + 1]:
                    idx_disp = idx_disp + 1

                if idx_disp + 1 < self.disp_stamps.shape[0] and \
                        np.abs(self.encoder_stamps[idx_encoder] - self.disp_stamps[idx_disp]) >= \
                        np.abs(self.encoder_stamps[idx_encoder] - self.disp_stamps[idx_disp + 1]):
                    idx_disp = idx_disp + 1

                while idx_rgbd + 1 < self.rgb_stamps.shape[0] and \
                        self.disp_stamps[idx_disp] >= self.rgb_stamps[idx_rgbd + 1]:
                    idx_rgbd = idx_rgbd + 1

                if idx_rgbd + 1 < self.rgb_stamps.shape[0] and \
                        np.abs(self.disp_stamps[idx_disp] - self.rgb_stamps[idx_rgbd]) >= \
                        np.abs(self.disp_stamps[idx_disp] - self.rgb_stamps[idx_rgbd + 1]):
                    idx_rgbd = idx_rgbd + 1

            tao = self.encoder_stamps[idx_encoder] - self.encoder_stamps[idx_encoder - 1]
            s_t = np.mean(self.encoder_counts[:, idx_encoder]) * 0.0022
            w_t = self.imu_angular_velocity[2, idx_imu]
            if w_t > 0.3:
                xs = np.arange(-4, 5)
                ys = np.arange(-4, 5)
            else:
                xs = np.arange(-4, 5)
                ys = np.arange(-4, 5)

            if idx_imu + 1 < self.imu_stamps.shape[0]:
                w_t = (self.imu_angular_velocity[2, idx_imu] + self.imu_angular_velocity[2, idx_imu + 1]) / 2

            s_random = np.random.normal(self.input_noise[0][0], np.abs(s_t) * self.input_noise[0][1], (1, self.num_ptc)) + s_t
            w_random = np.random.normal(self.input_noise[1][0], np.abs(w_t) * self.input_noise[1][1], (1, self.num_ptc)) + w_t

            s_tmp = s_random * self.sinc(w_random * tao / 2)

            state = state + np.vstack((s_tmp * np.cos(state[2, :] + w_random * tao / 2),
                                       s_tmp * np.sin(state[2, :] + w_random * tao / 2),
                                       w_random * tao))

            ranges = self.lidar_ranges[:, idx_lidar]
            indValid = np.logical_and((ranges < self.lidar_range_max), (ranges > self.lidar_range_min))

            ranges = ranges[indValid]
            angles = self.lidar_angles[indValid]
            body_detect = self.lidar2body(ranges, angles)

            if self.angle_shift and self.particle_shift:
                map_corr = []
                xy_is_lst = []
                for idx_theta in range(theta_s.shape[0]):
                    state_tmp = state.copy()
                    state_tmp[2, :] = state_tmp[2, :] + theta_s[idx_theta]

                    world_detect = self.body2world(body_detect, state_tmp)  # 2*1081*100

                    xy_is = np.ceil((world_detect - np.array([self.MAP['xmin'], self.MAP['ymin']]).reshape(2, 1, 1)) /
                                    self.MAP['res']).astype(np.int32) - 1
                    xy_is_lst.append(xy_is)
                    map_corr.append(self.map_correlation_mat_version(ranges, xy_is, xs=xs, ys=ys))

                xy_is = np.stack(xy_is_lst, axis=-1)  # 2*1081*100*theta
                map_corr = np.stack(map_corr, axis=-1)  # 100 * xy * theta
                max_squeeze_theta = np.max(map_corr, axis=2)
                max_squeeze_xy = np.max(map_corr, axis=1)
                corr_maxs = np.max(max_squeeze_theta, axis=-1)  # 100 * 1
                print(np.max(corr_maxs))
                corr_maxs = corr_maxs - np.max(corr_maxs)
                theta_shift = np.argmax(max_squeeze_xy, axis=-1)

            else:
                world_detect = self.body2world(body_detect, state)  # 2*1081*100

                xy_is = np.ceil((world_detect - np.array([self.MAP['xmin'], self.MAP['ymin']]).reshape(2, 1, 1)) /
                                self.MAP['res']).astype(np.int32) - 1

                max_squeeze_theta = self.map_correlation_mat_version(ranges, xy_is, xs=xs, ys=ys)  # 100 * 81
                corr_maxs = np.max(max_squeeze_theta, axis=1)  # 100 * 1
                print(np.max(corr_maxs))
                corr_maxs = corr_maxs - np.max(corr_maxs)

            particle_best_corrs = np.argmax(max_squeeze_theta, axis=1)  # 100 * 1 best shift for each article

            # weight update
            weight = weight * np.exp(corr_maxs).reshape(1, self.num_ptc)
            weight = weight / np.sum(weight)

            best_weight_idx = np.where(weight == np.max(weight))[1]
            if best_weight_idx.shape[0] > 0:
                best_weight_idx = np.random.choice(best_weight_idx)

            # particle shift
            if self.particle_shift and idx_encoder % 5 == 0:
                if self.angle_shift:
                    y_shift = np.floor_divide(particle_best_corrs, xs.shape[0])
                    x_shift = np.remainder(particle_best_corrs, ys.shape[0])

                    state[0, :] = state[0, :] + xs[x_shift] * self.MAP['res']
                    state[1, :] = state[1, :] + ys[y_shift] * self.MAP['res']
                    state[2, :] = state[2, :] + theta_s[theta_shift]

                    xy_is_best = xy_is[:, :, best_weight_idx, theta_shift[best_weight_idx]].reshape(2, -1) + np.vstack((xs[x_shift[best_weight_idx]],
                                                                                      ys[y_shift[best_weight_idx]]))
                else:
                    y_shift = np.floor_divide(particle_best_corrs, xs.shape[0])
                    x_shift = np.remainder(particle_best_corrs, ys.shape[0])

                    state[0, :] = state[0, :] + xs[x_shift] * self.MAP['res']
                    state[1, :] = state[1, :] + ys[y_shift] * self.MAP['res']
                    xy_is_best = xy_is[:, :, best_weight_idx].reshape(2, -1) + np.vstack((xs[x_shift[best_weight_idx]],
                                                                                          ys[y_shift[best_weight_idx]]))
            else:
                xy_is_best = xy_is[:, :, best_weight_idx].reshape(2, -1)

            trajectory = np.hstack((trajectory, state[:, best_weight_idx].reshape(-1, 1)))
            body_lidar = self.lidar2body(0, 0)
            world_lidar = self.body2world(body_lidar, state[:, best_weight_idx].reshape(-1, 1)).squeeze(2)  # 2 * 1 * 1
            lidar_is = np.ceil((world_lidar - np.array([self.MAP['xmin'], self.MAP['ymin']]).reshape(2, 1)) /
                               self.MAP['res']).astype(np.int32) - 1

            xy_occupied = np.hstack((xy_is_best, lidar_is))
            xy_occupied_idx = np.vstack((xy_occupied[1, :], xy_occupied[0, :])).T
            xy_occupied_idx = np.expand_dims(xy_occupied_idx, 0)
            self.MAP['log_map'][xy_is_best[0, :], xy_is_best[1, :]] += 2 * np.log(self.map_update_val)  # may invert

            mask_map = np.zeros_like(self.MAP['map'])
            cv2.drawContours(image=mask_map, contours=[xy_occupied_idx], contourIdx=0,
                             color=np.log(1.0 / self.map_update_val), thickness=-1)

            self.MAP['log_map'] += mask_map

            occupied = self.MAP['log_map'] > self.draw_threshold
            empty = self.MAP['log_map'] <= self.draw_threshold
            self.MAP['map'][occupied] = 1
            self.MAP['map'][empty] = 0

            # camera
            if self.do_texture:
                depth, rgbi, rgbj = self.read_disparity(idx_disp)
                rgb_image = self.read_rgb_image(idx_rgbd)

                robot_pose = state[:2, best_weight_idx].reshape(-1, 1)
                robot_ori = state[2, best_weight_idx]
                robot_pose = np.vstack((robot_pose, np.array([[0.127]])))
                rotation_camera_body2world = np.array([[np.cos(robot_ori), -np.sin(robot_ori), 0],
                                                      [np.sin(robot_ori), np.cos(robot_ori), 0],
                                                      [0, 0, 1]])
                transform_camera_body2world = np.hstack((rotation_camera_body2world, robot_pose))
                transform_camera_body2world = np.vstack((transform_camera_body2world, np.array([[0, 0, 0, 1]])))

                transform_camera_cam2body = self.transform_cam2body

                pixels = np.vstack((rgbi.reshape(1, -1), rgbj.reshape(1, -1), np.ones_like(rgbi.reshape(1, -1))))
                position_optical = depth.reshape(1, -1) * np.dot(self.inv_camera_calibration, pixels)
                position_optical = np.vstack((position_optical, np.ones_like(depth.reshape(1, -1))))

                transform_optical2camera = np.hstack((self.inv_camera_roc, np.zeros((3, 1))))
                transform_optical2camera = np.vstack((transform_optical2camera, np.array([[0, 0, 0, 1]])))
                position_camera = np.dot(transform_optical2camera, position_optical)

                position_world = np.dot(np.dot(transform_camera_body2world,
                                               transform_camera_cam2body),
                                        position_camera).T  # num_pixels * 4

                height_threshold = self.height_threshold
                pixel_indValid = np.logical_and(position_world[:, 2] >= height_threshold[0],
                                                position_world[:, 2] <= height_threshold[1])

                position_world = position_world[pixel_indValid, :]
                camera_xy_is = np.ceil(
                    (position_world[:, :2].T - np.array([self.MAP['xmin'], self.MAP['ymin']]).reshape(2, 1)) /
                    self.MAP['res']).astype(np.int32) - 1

                rgbi = rgbi[pixel_indValid]
                rgbj = rgbj[pixel_indValid]
                self.MAP['texture_map'][camera_xy_is[0, :], camera_xy_is[1, :], :] = rgb_image[rgbj, rgbi, :]

                # img_tmp = np.zeros((self.MAP['sizex'], self.MAP['sizey'], 3))

                # img_tmp = np.zeros_like(self.MAP['log_map'])
                # img_tmp[self.MAP['log_map'] <= -self.draw_threshold] = 1

                self.MAP['texture_map'] = self.MAP['texture_map'] * \
                                          (self.MAP['log_map'] <= -self.draw_threshold).reshape(self.MAP['sizex'],
                                                                                                self.MAP['sizey'],
                                                                                                1)

                # xy_val = dict()
                # for idx_pixel in range(camera_xy_is.shape[1]):
                #     key = (camera_xy_is[0, idx_pixel], camera_xy_is[1, idx_pixel])
                #     if tuple in xy_val:
                #         xy_val[key] = xy_val[key] + rgb_image[rgb]

            if idx_encoder % 50 == 0:
                print(idx_encoder)
                self.save_img(trajectory, idx_encoder)

            # resample
            Neff = 1 / np.sum(weight * weight)
            if(Neff / self.num_ptc) < self.resample_threshold:
                # print("get in resample")
                j_resample = 0
                c_resample = weight[0, j_resample]
                N_resample = self.num_ptc
                state_tmp = np.zeros_like(state)
                for k in range(N_resample):
                    miu = np.random.uniform(0, 1/N_resample)
                    beta = miu + (k - 1) / N_resample
                    while beta > c_resample:
                        j_resample = j_resample + 1
                        c_resample = c_resample + weight[0, j_resample]

                    state_tmp[:, k] = state[:, j_resample]
                weight = np.ones((1, self.num_ptc)) / self.num_ptc
                state = state_tmp

        print(idx_encoder)
        self.save_img(trajectory, idx_encoder)
        return trajectory

    def save_img(self, trajectory, idx_encoder):
        img_tmp = np.zeros((self.MAP['sizex'], self.MAP['sizey'], 3))
        draw_threshold = self.draw_threshold
        img_tmp[self.MAP['log_map'] > draw_threshold, :] = (255, 255, 255)
        img_tmp[self.MAP['log_map'] < -draw_threshold, :] = (125, 125, 125)
        trajectory_tmp = np.ceil((trajectory[:2, :] - np.array([self.MAP['xmin'], self.MAP['ymin']]).reshape(2, 1)) /
                self.MAP['res']).astype(np.int32) - 1

        trajectory_tmp = np.vstack((trajectory_tmp[1, :], trajectory_tmp[0, :])).T
        trajectory_tmp = trajectory_tmp.reshape((-1, 1, 2)).astype(np.int32)

        img_tmp = cv2.polylines(img_tmp, [trajectory_tmp], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.imwrite(self.dir_name + '/map_at_{}.png'.format(idx_encoder), img_tmp)

        if self.do_texture:
            img_tmp = np.zeros((self.MAP['sizex'], self.MAP['sizey'], 3))
            img_tmp[self.MAP['log_map'] > draw_threshold, :] = (255, 255, 255)
            img_tmp[self.MAP['log_map'] < -draw_threshold, :] = (125, 125, 125)

            img_tmp[np.sum(self.MAP['texture_map'], axis=2) != 0] = (0, 0, 0)
            img_tmp = img_tmp + self.MAP['texture_map']
            trajectory_tmp = np.ceil((trajectory[:2, :] - np.array([self.MAP['xmin'], self.MAP['ymin']]).reshape(2, 1)) /
                                     self.MAP['res']).astype(np.int32) - 1

            trajectory_tmp = np.vstack((trajectory_tmp[1, :], trajectory_tmp[0, :])).T
            trajectory_tmp = trajectory_tmp.reshape((-1, 1, 2)).astype(np.int32)

            img_tmp = cv2.polylines(img_tmp, [trajectory_tmp], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.imwrite(self.dir_name + '/texture_at_{}.png'.format(idx_encoder), img_tmp)


    def map_correlation_mat_version(self, ranges, vp, xs=np.arange(-4, 5), ys=np.arange(-4, 5)):
        """
        :param vp: 2*1081*N, represent xy, angle, state_n
        :param xs: x shift, a numpy array, arange from left shift to right shift np.arange(-4, 5)
        :param ys: y shift, a numpy array, arange from left shift to right shift np.arange(-4, 5)
        :return: N*81 np array.
        """
        x_vp = vp[0, :, :]
        y_vp = vp[1, :, :]

        x_vp_shape = x_vp.shape  # (1081, 100)
        x_vp = x_vp.reshape(-1, 1)  # prior n particle, then 1081 angle
        y_vp = y_vp.reshape(-1, 1)  # prior n particle, then 1081 angle

        y_change, x_change = np.where(np.ones((ys.shape[0], xs.shape[0])) == 1)
        x_change = (x_change + xs[0]).astype(np.int32)  # change fast
        y_change = (y_change + ys[0]).astype(np.int32)  # change slow

        x_vp = np.dot(x_vp, np.ones((1, x_change.shape[0]), dtype=np.int32)) + x_change.reshape(1, -1)
        y_vp = np.dot(y_vp, np.ones((1, y_change.shape[0]), dtype=np.int32)) + y_change.reshape(1, -1)

        flag = np.logical_and(np.logical_and(x_vp >= 0, x_vp < self.MAP['sizex']),
                              np.logical_and(y_vp >= 0, y_vp < self.MAP['sizey']))

        x_vp[np.logical_not(flag)] = 0
        y_vp[np.logical_not(flag)] = 0

        # _, map_mask = cv2.threshold(self.MAP['map'] * 255, 127, 1, cv2.THRESH_BINARY)
        # img_tmp = cv2.dilate(map_mask, np.ones((2, 2), np.uint8)) + self.MAP['map']
        img_tmp = self.MAP['map']
        img_extract = img_tmp[x_vp, y_vp]
        # img_extract = self.MAP['map'][x_vp, y_vp]
        img_extract[np.logical_not(flag)] = 0

        img_extract = img_extract.reshape(x_vp_shape[0], x_vp_shape[1], -1)  # (1081, 100, 81)
        corr_weight = np.power(ranges, 2.5)
        img_extract = img_extract * corr_weight.reshape(ranges.shape[0], 1, 1)
        img_extract = np.sum(img_extract, axis=0)  # (100, 81)

        return img_extract

    def lidar2body(self, range, angles):
        x = range * np.cos(angles)
        y = range * np.sin(angles)
        res = np.vstack((x, y)) + self.lidar_position
        return res

    def rotation_mat(self, theta):
        """
        :param theta: robot orientation, from north to orientation. 1*N
        :return: Rotation matrix from body to world 2*2*N
        """
        r11 = np.expand_dims(np.cos(theta), 0)
        r12 = np.expand_dims(-np.sin(theta), 0)
        r21 = np.expand_dims(np.sin(theta), 0)
        r22 = np.expand_dims(np.cos(theta), 0)

        return r11, r12, r21, r22

    def body2world(self, pos, state):
        """
        :param pos: 2 * 1080 array, representing x, y position in body frame
        :param state: (3, ) array, representing robot current state (x, y, orientation)
        :return: 2 * N array, representing x, y positions in world frame
        """
        r11, r12, r21, r22 = self.rotation_mat(state[2, :])

        level1 = np.dot(pos[0, :].reshape(-1, 1), r11) + np.dot(pos[1, :].reshape(-1, 1), r12)
        level2 = np.dot(pos[0, :].reshape(-1, 1), r21) + np.dot(pos[1, :].reshape(-1, 1), r22)

        res = np.stack((level1, level2), axis=0)
        res = res + np.expand_dims(state[:2, :], 1)
        return res


    def pre_trajectory(self):
        state = np.zeros((3, 1))
        idx_imu = 0
        trajectory = np.zeros((3, 0))
        for idx_encoder in range(1, self.encoder_stamps.shape[0]):
            while idx_imu + 1 < self.imu_stamps.shape[0] and\
                    self.encoder_stamps[idx_encoder] >= self.imu_stamps[idx_imu + 1]:
                idx_imu = idx_imu + 1

            tao = self.encoder_stamps[idx_encoder] - self.encoder_stamps[idx_encoder - 1]
            s_t = np.mean(self.encoder_counts[:, idx_encoder]) * 0.0022
            if idx_imu + 1 < self.imu_stamps.shape[0]:
                w_t = (self.imu_angular_velocity[2, idx_imu] + self.imu_angular_velocity[2, idx_imu + 1]) / 2
            else:
                w_t = self.imu_angular_velocity[2, idx_imu]
            s_t = s_t * np.ones((1, 1))
            w_t = w_t * np.ones((1, 1))
            s_tmp = s_t * self.sinc(w_t * tao / 2)
            state = state + np.vstack((s_tmp * np.cos(state[2, :] + w_t * tao / 2),
                                      s_tmp * np.sin(state[2, :] + w_t * tao / 2),
                                      w_t * tao))
            # print("state", state)
            trajectory = np.hstack((trajectory, state.reshape(3, -1)))

        return trajectory

    def sinc(self, x):
        return np.divide(np.sin(x), x)

    def draw_pre_trajectory(self):
        plt.figure()
        plt.plot(self.trajectory_pre[0, :], self.trajectory_pre[1, :])
        plt.show()


if __name__ == '__main__':
    ts = tic()
    vel = MyVehicle(dataset=20, num_ptc=150, do_texture=True, angle_shift=False, particle_shift=True, name="particle_shift_range0")
    vel.run_main()
    toc(ts, "dataset 21 using")

    ts = tic()
    vel = MyVehicle(dataset=20, num_ptc=150, do_texture=True, angle_shift=False, particle_shift=True, name="particle_shift")
    vel.run_main()
    toc(ts, "dataset 20 using")

    # ts = tic()
    # vel = MyVehicle(dataset=23, num_ptc=150, do_texture=False, angle_shift=False, particle_shift=True, name="range2p5")
    # vel.run_main()
    # toc(ts, "dataset 23 using")
