"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)
"""

import math

import matplotlib.pyplot as plt
import numpy as np

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
alphas=np.array([0.11, 0.01, 0.18, 0.08, 0.0, 0.0])

#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

# DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIFF_TH = 1.6
M_DIST_TH = 2
# M_DIST_TH_FIRST = 0.25  # Threshold of Mahalanobis distance for data association.
M_DIST_TH_ALL = 1
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]


show_animation = True
    
def ekf_slam(xEst, PEst, u, z, dt, logger, xTruth, obs_cones_w_idx):
    cones = []
    # Predict
    S = STATE_SIZE
    # logger.info(f'{xEst[0:S]}')
    logger.info(f'{u}')
    logger.info(f"dt: {dt}")
    G, Fx = jacob_motion(xEst[0:S], u, dt, logger)

    #calculate motion covariance in control space
    M_t = np.array([[(alphas[0] * abs(u[0, 0]) + alphas[1] * abs(u[1, 0]))**2, 0],
                        [0, (alphas[2] * abs(u[0, 0]) + alphas[3] * abs(u[1, 0]))**2]])
    
    x = xEst[0:S]
    #calculate Jacobian to transform motion covariance to state space 
    V_t = np.array([[np.cos(x[2, 0]), -0.5 * np.sin(x[2, 0])],
                    [np.sin(x[2, 0]), 0.5 * np.cos(x[2, 0])],
                    [0, 1]])
    logger.info(f'M_t: {M_t}')
    logger.info(f'V_t: {V_t}')
    # Cx = V_t @ M_t @ V_t.T
    xEst[0:S] = motion_model(xEst[0:S], u, dt)
    PEst[0:S, 0:S] = G.T @ PEst[0:S, 0:S] @ G + Fx.T @ Cx @ Fx
    initP = np.eye(2)
    logger.info(f'Motion Model: {xEst[0, 0]}, {xEst[1, 0]}')

    
    min_id_errors = 0
    # Update
    print("Num observations: ", len(z))
    print("Ground truth num obs: ", len(obs_cones_w_idx))
    for iz in range(len(z[:, 0])):  # for each observed cone 
        #iz is the index of a particular cone 
        min_id = search_correspond_landmark_id(xEst, PEst, z[iz, 0:2], logger, xTruth)
        cones.append(calc_landmark_position(xEst, z[iz, :]))
        nLM = calc_n_lm(xEst)
        
        # Cases for min_id
        # 1.) min_id = nLM (the last element)
        # This happens because the last element appended is the threshold distance
        # this means no small distance found smaller than threshold
        # 
        # 2.) min_id = some other index

# HEURISTICS STARTS HERE
# preconditions: 
# cones in z must be in the same order as self.cones
# for loop must be iterating through each observation 

#Issue: need to set min_id to the correct one
#Need to also track the idx of old cones in xEst
# Possible solution: make a tuple, each tuple has xTruth_cone_idx, true/false

#obs_cones_w_idx is a 2d list
# first column represents the cone_idx
# second column tells whether new cone or not
        found_in_xTruth = False
        is_new = False
        for obs in obs_cones_w_idx:
            if obs[0] == min_id:
                found_in_xTruth = True
                is_new = obs[1]
                break
        # min_id calculated wrong 
        #Case 1: not found
        if not found_in_xTruth:
            # correcting min_id
            min_id = obs_cones_w_idx[iz][0]
            min_id_errors += 1
        

        if is_new:
            if min_id != nLM:
                min_id_errors += 1
                min_id = obs_cones_w_idx[iz][0]
            print("New LM")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug
            PEst = PAug
        else:
            if min_id != obs_cones_w_idx[iz][0]:
                min_id_errors += 1
                min_id = obs_cones_w_idx[iz][0]
        min_id = int(min_id)

        # if min_id == nLM: #supposedly new landmark

        #     if is_new:
        #         print("New LM")
        #         # Extend state and covariance matrix
        #         xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
        #         PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
        #                           np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
        #         xEst = xAug
        #         PEst = PAug
        #     else: #if is_new false, then min_id shouldn't be nLM(decision incorrect) 
        #         #min_id calculated wrong
        #         data_association_errors += 1
        #         
        #         
        # elif min_id != nLM: #supposedly not new landmark
        #     if is_new: #if iz in new_cones_idx, then new landmark (decision incorrect)
        #         data_association_errors += 1
        #         print("New LM")
        #         # Extend state and covariance matrix
        #         xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
        #         PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
        #                           np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
        #         xEst = xAug
        #         PEst = PAug

        #         # correcting min_id
        #         min_id = nLM
        #         min_id_errors += 1
        #     # if !is_new, then make sure min_id is correct

        # 


        # print("type of min_id: ", type(min_id))    
        # min_id starts at 0
        lm = get_landmark_position_from_state(xEst, min_id)
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

        K = (PEst @ H.T) @ np.linalg.inv(S)
        # logger.info(f'Kalman: {K}')
        # logger.info(f'xEST: {xEst.shape}')
        # logger.info(f'K: {K.shape}')
        # logger.info(f'y: {y.shape}')

        #updating the position of the landmarks?
        xEst[3:, :] = xEst[3:, :] + (K[3:, :] @ y)
        # xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst
    # logger.info(f'Landmark Update: {xEst[0, 0]}, {xEst[1, 0]}')
    # logger.info(f'\n\n')
    xEst[2] = pi_2_pi(xEst[2])
    #PEst = pose estimate
    #xEst = Robot pose estimate 

    #xEst is done
    #Compare xEst with xTruth

    #Calculate the error in the 
    # for i in range(3, len(xTruth), 2):
    #     err = math.sqrt((xTruth[i][0] - xEst[i][0])**2 + (xTruth[i+1][0] - xEst[i+1][0])**2 ) 
    #     xErr[(i - 3)/2] = err




    return xEst, PEst, cones,  min_id_errors


def calc_input():
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_n = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise
            zi = np.array([dn, angle_n, i])
            z = np.vstack((z, zi))

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, ud)
    print(z)
    return xTrue, z, xd, ud


def motion_model(x, u, dt):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[dt * math.cos(x[2, 0]), 0],
                  [dt * math.sin(x[2, 0]), 0],
                  [0.0, dt]])

    x = (F @ x) + (B @ u)
    return x


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x, u, dt, logger):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_lm(x)))))
    # logger.info(f'{x.shape}')
    jF = np.array([[0.0, 0.0, -dt * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, dt * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx

    return G, Fx,


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x, ind):
    print("Length of xEst: ", len(x))
    print("Index: ", ind)
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm



def search_correspond_landmark_id(xAug, PAug, zi, logger, xTruth):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xAug)
    min_dist = []
    # min_dist, second_min_dist = [1000, -1], [1000, -1] #[mahalanobis, id]

    # This is the range and bearing of the observed landmark 
    r, theta = zi[0], zi[1]
    # logger.info(f'Measurement: {r*math.cos(theta)}, {r*math.sin(theta)}')
    for i in range(nLM): #iterate through all of the previously seen landmarks
        lm = get_landmark_position_from_state(xAug, i)
        print("Length of truth table: ", len(xTruth))
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        # mahalanobis = y.T @ np.linalg.inv(S) @ y
        # if mahalanobis < min_dist[0]:
        #     second_min_dist = min_dist
        #     min_dist = [mahalanobis, i]
        # elif mahalanobis < second_min_dist[0]:
        #     second_min_dist = [mahalanobis, i]
        # else:
        #     continue
        min_dist.append(y.T @ np.linalg.inv(S) @ y)
        # logger.info(f'  Landmark {i}: {lm[0]}, {lm[1]} | Mahalanobis: {y.T @ np.linalg.inv(S) @ y}')

    min_dist.append(M_DIST_TH)  # new landmark; minimum distance threshold??
    min_id = min_dist.index(min(min_dist))

#issue: why are we checking mahalanobis distance for new cones with xTruth?
# we have already appended the new cones to xTruth
# 

#HEURISTIC STARTS HERE
    # min_dist_truth = []
    # for i in range(nLM):
    #     lm = get_landmark_position_from_state(xTruth, i)
    #     y, S, H = calc_innovation(lm, xTruth, PAug, zi, i)
    #     min_dist_truth.append(y.T @ np.linalg.inv(S) @ y)

    # min_dist_truth.append(M_DIST_TH)  # new landmark
    # min_id_truth = min_dist_truth.index(min(min_dist_truth))

    # if min_id != nLM:      #  found existing landmark
    #     if min_id_truth == nLM: # mindist calc is really bad
    #         error += 1
    #         return nLM
    # else: # didnt find existing landmark
    #     if min_id_truth != nLM: # should have found landmark
    #         error += 1
    #         return min_id_truth
    # 

    # logger.info(f'   {second_min_dist[0]}/{min_dist[0]} == {second_min_dist[0]/min_dist[0]} ')
    # if nLM == 0:
    #     return 0
    # elif nLM == 1:
    #     return 0 if min_dist[0] < M_DIST_TH_FIRST else 1
    # else:
    #     return min_dist[1] if second_min_dist[0]/min_dist[0] > M_DIFF_TH else nLM
    return min_id

    

def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + Cx[0:2, 0:2]

    return y, S, H


def jacob_h(q, delta, x, i):
    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0],
                     [-7.5, 7.5],
                     [-2.1, 10.2]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
