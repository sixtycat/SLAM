import numpy as np
from scipy import linalg
from starter_code.utils import *
import matplotlib.pyplot as plt


def adjoint_se3(vector):
    assert vector.shape == (6, 1)
    v1 = vector[:3].reshape(3, 1)  # v
    v2 = vector[3:].reshape(3, 1)  # w
    v1_hat = vector2so3(v1)
    v2_hat = vector2so3(v2)
    adjoint = np.vstack((np.hstack((v2_hat, v1_hat)),
                         np.hstack((np.zeros((3, 3)), v2_hat))))
    return adjoint


def inv_T(matrix_input):
    assert matrix_input.shape == (4, 4)
    R = matrix_input[:3, :3]
    p = matrix_input[:3, 3]
    res = np.zeros((4, 4))
    res[:3, :3] = R.T
    res[:3, 3] = -R.T @ p
    res[3, 3] = 1
    return res


def vector2so3(vector):
    assert vector.shape == (3, 1)
    return np.array([[0, -vector[2, 0], vector[1, 0]],
                     [vector[2, 0], 0, -vector[0, 0]],
                     [-vector[1, 0], vector[0, 0], 0]])


def vector2se3(vector):
    assert vector.shape == (6, 1)
    res1 = vector[:3].reshape(3, 1)
    res2 = vector[3:].reshape(3, 1)
    res = np.vstack((np.hstack((vector2so3(res2), res1)),
                     np.zeros((1, 4))))
    return res


def se32vector(x):
    assert x.shape == (4, 4)
    x_tmp1 = x[:3, 3].reshape(-1, 1)
    x_tmp2 = np.array([[x[2, 1]], [x[0, 2]], [x[1, 0]]])
    return np.vstack((x_tmp1, x_tmp2))


def circle_dot(vector):
    assert vector.shape == (4, 1)
    s = vector[:3].reshape(3, 1)
    lam = vector[3]
    return np.vstack((np.hstack((lam * np.eye(3), -vector2so3(s))),
                      np.zeros((1, 6))))


def pixel2world(pixel, K, b, cam_T_world):
    assert pixel.shape == (4, 1)
    z = K[0, 0] * b / (pixel[0, 0] - pixel[2, 0])
    x = z * (pixel[0, 0] - K[0, 2]) / K[0, 0]
    y = z * (pixel[1, 0] - K[1, 2]) / K[1, 1]

    X_o = np.array([x, y, z, 1])
    X_w = np.linalg.inv(cam_T_world) @ X_o
    X_w = np.divide(X_w, X_w[-1])
    return X_w


# def init_landmark(landmark, feature, K, b, cam_T_world):
#     for i in range(landmark.shape[1]):
#         if landmark[3, i] == 0:
#             if np.sum(np.abs(feature[:, i] + 1)) != 0:
#                 landmark[:, i] = pixel2world(feature[:, i].reshape(-1, 1), K, b, cam_T_world)

def init_landmark(landmarks, feature, idx_valids, K, b, cam_T_world):
    for i in range(idx_valids.shape[0]):
        if landmarks[3, idx_valids[i]] == 0:
            if np.sum(np.abs(feature[:, idx_valids[i]] + 1)) != 0:
                landmarks[:, idx_valids[i]] = pixel2world(feature[:, idx_valids[i]].reshape(-1, 1), K, b, cam_T_world)


def pi_func(vector):
    """
    :param vector: 4 * N
    :return: 4 * N
    """
    assert len(vector.shape) == 2
    assert vector.shape[0] == 4
    return np.divide(vector, vector[2, :].reshape(1, -1))

def func_dpi_dq(vector):
    dpi_dq = np.expand_dims(np.eye(4), axis=2) * np.ones((4, 4, vector.shape[1]))
    dpi_dq[:, 2, :] -= pi_func(vector)
    dpi_dq = dpi_dq / vector[2, :].reshape(1, -1)
    return dpi_dq


if __name__ == '__main__':
    filename = "./data/0042.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)

    s_imu = 0.01
    s_Li = 0
    s_LM = 10000
    V = 1000
    s_W = 0.0000001

    num_LM = features.shape[1]
    Sigma_both = np.vstack((np.hstack((s_imu * np.eye(6), s_Li * np.ones((6, 3 * num_LM)))),
                       np.hstack((s_Li * np.ones((3 * num_LM, 6)), s_LM * np.eye(3 * num_LM)))))

    W = np.zeros((6 + 3 * features.shape[1], 6 + 3 * features.shape[1]))
    W[:6, :6] = np.eye(6) * s_W

    K_inv = np.linalg.inv(K)

    input_u = np.vstack((linear_velocity, rotational_velocity))
    M_cam = np.array([[K[0, 0], 0, K[0, 2], 0],
                      [0, K[1, 1], K[1, 2], 0],
                      [K[0, 0], 0, K[0, 2], -K[0, 0] * b],
                      [0, K[1, 1], K[1, 2], 0]])  # intrincic matrix

    D = np.vstack((np.eye(3), np.zeros((1, 3))))

    miu = np.zeros((6 + 4 * num_LM, 1))
    imu_T_cam = inv_T(cam_T_imu)

    result_c_T = list()
    result_c_T.append(inv_T(linalg.expm(vector2se3(miu[:6]))))

    Fx = np.zeros((9, 6 + 3 * num_LM))
    Fx[:6, :6] = np.eye(6)

    # initial land mark at 0
    iTw = linalg.expm(vector2se3(miu[:6]))
    landmark = miu[6:].reshape(-1, 4).T
    landmark_init = np.zeros_like(landmark)
    idx_valid = np.where(np.sum(np.abs(features[:, :, 0] + 1), axis=0) > 0)[0]
    init_landmark(landmark, features[:, :, 0], idx_valid, K, b, cam_T_imu @ iTw)
    init_landmark(landmark_init, features[:, :, 0], idx_valid, K, b, cam_T_imu @ iTw)
    print("initial done")

    for idx_t in range(t.shape[1] - 1):
        if idx_t % 100 == 0:
            print(idx_t)

        tao = t[0, idx_t + 1] - t[0, idx_t]

        # step 0 prediction
        ut = input_u[:, idx_t].reshape(-1, 1)
        iTw = linalg.expm(vector2se3(miu[:6]))
        iTw = linalg.expm(-tao * vector2se3(ut)) @ iTw
        miu[:6] = se32vector(linalg.logm(iTw))

        Gt = np.eye(6 + 3 * num_LM)
        Gt[:6, :6] = linalg.expm(-tao * adjoint_se3(ut))
        Sigma_both = Gt @ Sigma_both @ Gt.T + tao ** 2 * W

        cam_T_world = cam_T_imu @ iTw
        landmark = miu[6:].reshape(-1, 4).T

        feature_t = features[:, :, idx_t + 1]
        idx_valid = np.where(np.sum(np.abs(feature_t + 1), axis=0) > 0)[0]
        init_landmark(landmark, feature_t, idx_valid, K, b, cam_T_world)
        init_landmark(landmark_init, feature_t, idx_valid, K, b, cam_T_world)
        miu[6:] = landmark.T.reshape(-1, 1)
        z_t = feature_t[:, idx_valid].reshape(4, -1)
        Nt = idx_valid.shape[0]

        if Nt == 0:
            continue

        landmark = miu[6:].reshape(-1, 4).T
        landmark_valid = landmark[:, idx_valid].reshape(4, -1)

        pi_input = cam_T_imu @ iTw @ landmark_valid
        z_hat = M_cam @ pi_func(pi_input)
        Mdpi_dq = M_cam @ func_dpi_dq(pi_input)

        Ht = np.zeros((4 * Nt, 6 + 3 * num_LM))

        for idx_H in range(Nt):
            Fx_tmp = Fx
            Fx_tmp[6:, 6 + 3 * idx_valid[idx_H]: 6 + 3 * (idx_valid[idx_H] + 1)] = np.eye(3)
            Ht_imu = Mdpi_dq[:, :, idx_H] @ cam_T_imu @ circle_dot(iTw @ landmark_valid[:, idx_H].reshape(4, -1))
            Ht_LM = Mdpi_dq[:, :, idx_H] @ cam_T_world @ D
            Ht_i = np.hstack((Ht_imu, Ht_LM)) @ Fx_tmp
            Ht[4 * idx_H: 4 * (idx_H + 1), :] = Ht_i

        Kt = Sigma_both @ Ht.T @ np.linalg.inv(Ht @ Sigma_both @ Ht.T + V * np.eye(4 * Nt))
        miu_update = Kt @ ((z_t - z_hat).T.reshape(-1, 1))
        miu_LM = np.vstack((miu_update[6:].reshape(-1, 3).T, np.zeros((1, num_LM))))

        miu = miu + np.vstack((miu_update[:6], miu_LM.T.reshape(-1, 1)))
        Sigma_both = (np.eye(6 + 3 * num_LM) - Kt @ Ht) @ Sigma_both

        iTw = linalg.expm(vector2se3(miu[:6]))
        result_c_T.append(inv_T(iTw))

    result_c_T = np.stack(result_c_T, axis=2)
    landmark = miu[6:].reshape(-1, 4).T
    fig1, ax1 = visualize_trajectory_2d(result_c_T, path_name="path", show_ori=True)
    ax1.plot(landmark_init[0, :], landmark[1, :], '.b', label="initialization", markersize=5)
    ax1.plot(landmark[0, :], landmark[1, :], '.y', label="landmark", markersize=5)
    ax1.legend()
    plt.show()