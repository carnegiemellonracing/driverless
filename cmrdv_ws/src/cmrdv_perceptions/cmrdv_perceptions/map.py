import numpy as np

"""
Contains Map class for the map of landmarks and current robot position 

Accounts for data association when seeing a new observation 
and uknown number of landmarks at beginning 

Translated from MATLAB: http://andrewjkramer.net/intro-to-the-ekf-step-4/
Documentation and structure based on: https://github.com/Attila94/EKF-SLAM
"""

class Map:
    """
    Class for map containing landmarks and current robot position   

    Attributes
    ----------
    alphas: numpy array 
        array of motion model noise parameters 
    Q_t: numpy array 
        array of measurement model noise parameters 
    state_mean: numpy array 
        mean of map (containing landmarks and current robot position), state vector of length 3 + (2 * n_landmarks) 
        initial state_mean defined as [x, y, heading] when observed landmarks is 0 
        state_mean will increase in length by 2 every time we add a new landmark to the map: append [x, y] of the landmark to end of array 
    state_cov: numpy array 
        covariance of all the landmarks and current robot position 
        covariance matrix of 3 + (2 * n_landmarks) by 3 + (2 * n_landmarks) with same structure as state_mean 
        first 3x3 submatrix are the covariances of the robot position 
    n_landmarks: int
        number of unique landmarks we have observed so far 
    loop_closure: boolean 
        indicator of whether loop closure has occured; when the first landmark has been reobserved (assume we have completed one lap)

    current_cone_left: int
        index of current landmark our observation of left cone corresponds to 

    current_cone_right: int 
        index of current landmark our observation of right cone corresponds to

    updated_cone: bool
        indicated whether the observation we observed corresponds to a different/new landmark from previous observation 

    """
    # TODO: tune default alpha values
    def __init__(self,
                 alphas=np.array([0.11, 0.01, 0.18, 0.08, 0.0, 0.0]),
                 Q_t=np.array([[11.7, 0.0],
                               [0.0, 0.18]])):
        self.alphas = alphas 
        self.Q_t = Q_t 
        self.state_mean = np.zeros(3)
        self.state_cov = 0.001 * np.ones((3, 3))
        self.n_landmarks = 0
        self.loop_closure = False 

        self.current_cone_left = -1 
        self.current_cone_right = -1
        self.current_cone_left_color = -1
        self.current_cone_right_color = -1 
        # assuming 0 is left color, 1 is right color, 2 is orange color

        self.updated_cone = False 

        self.robot_state = np.zeros(3) #get the first 3 elements from state_mean vector
        self.prev_robot_state = np.zeros(3) #robot state from previous observation 
        self.curr_time = 0 #time of current robot state
        self.prev_time = 0 #time of previous robot state 

        self.d_robot_state = np.zeros(3) #changes in [x,y,heading] in robot state 
        self.d_time = 0 #change in time for robot 

    # odometry and measurement sample, update robot's pose 
    def __predict(self, movement, new_time):
        """Updates robot position given change in polar coordinates 

        Parameters 
        ----------
        movement: array 
            contains array of [new x coordinate, new y coordinate, new heading] 
        new_time: float 
            updated change in time since previous robot movement 
    
        Output
        ----------
        Updates self.state_mean and self.state_cov of the robot position (x, y, heading) 
        based on given changes in robot position 

        """
        # update self.d_time and self.d_robot_state 
        new_x = movement[0]
        new_y = movement[1]
        new_heading = movement[2] 

        # update previous states 
        self.prev_time = self.curr_time 
        self.prev_robot_state = np.copy(self.robot_state) 

        # updates current states 
        self.curr_time = new_time 
        self.robot_state = np.array((new_x, new_y, new_heading)) 

        # updates changes in position/time 
        self.d_time = self.curr_time - self.prev_time 
        self.d_robot_state = self.robot_state - self.prev_robot_state 


        dt = self.d_time 
        '''
        dt: int 
            change in time since previous robot movement 
        '''

        d_theta = self.d_robot_state[2] #taking 3rd element in robot state 
        d_trans = np.sqrt((self.d_robot_state[0]**2) + (self.d_robot_state[1]**2)) # sqrt(dx^2 + dy^2) 
        #change in translation/distance - Euclidean distance? 
        # QUESTION: is this absolute value?? would this always be magnitude? 

        u_t = [d_theta, d_trans]
        '''
        u_t: list or numpy array 
            movement vector: [change in theta/rotation, change in translation/distance]
        '''

        n = len(self.state_mean)
        theta = self.state_mean[2]
        dtheta = dt * u_t[1] #change in theta
        dhalf_theta = dtheta / 2
        dtrans = dt * u_t[0] #change in translation 

        #calculate pose update from odometry (motion model)
        pose_update = np.array([dtrans * np.cos(theta + dhalf_theta),
                                dtrans * np.sin(theta + dhalf_theta),
                                dtheta])

        #updated state mean 
        F_x = np.append(np.eye(3),np.zeros((3,n-3)),axis=1)

        state_mean_bar = self.state_mean + (F_x.T).dot(pose_update)
        state_mean_bar[2] = (state_mean_bar[2] + 2 * np.pi) % (2 * np.pi)

        #calculate movement Jacobian 
        g_t = np.array([[0,0,dtrans*-np.sin(theta + dhalf_theta)],
                        [0,0,dtrans*np.cos(theta + dhalf_theta)],
                        [0,0,0]])
        G_t = np.eye(n) + (F_x.T).dot(g_t).dot(F_x)

        #calculate motion covariance in control space 
        M_t = np.array([[(self.alphas[0] * abs(u_t[0]) + self.alphas[1] * abs(u_t[1]))**2, 0],
                        [0, (self.alphas[2] * abs(u_t[0]) + self.alphas[3] * abs(u_t[1]))**2]])

        #calculate Jacobian to transform motion covariance to state space 
        V_t = np.array([[np.cos(theta + dhalf_theta), -0.5 * np.sin(theta + dhalf_theta)],
                        [np.sin(theta + dhalf_theta), 0.5 * np.cos(theta + dhalf_theta)],
                        [0, 1]])


        #update state covariance 
        R_t = V_t.dot(M_t).dot(V_t.T)
        state_cov_bar = (G_t.dot(self.state_cov).dot(G_t.T)) + F_x.T.dot(R_t).dot(F_x)

        #updated state mean after the prediction 
        self.state_mean = state_mean_bar

        self.robot_state = self.state_mean[0:3]
        
        self.state_cov = state_cov_bar

    

    def __update(self, z):
        """Updates map of landmarks given potential observations 

        Parameters 
        ----------
        z: list of lists or 2D numpy array 
            list of potential measurements of landmarks
            each potential landmark is represented as a list of [x, y, color] from the current robot position 
            *****have to change to input cartesian coordinates (from the robot position?) 
            color indicates whether cone is left or right 

        Output
        ----------
        Updates self.state_mean, self.state_cov, self.n_landmarks with new updated positions of landmarks 
        If loop closure occured, then self.loop_closure will be True 

        """
        # update change in time since its already been updated in predict
        dt = self.d_time 

        #z: potential measurements of landmarks 
        state_mean_bar = self.state_mean
        state_cov_bar = self.state_cov
        n_landmarks = self.n_landmarks

        # we don't know if we will update the landmark we are looking at yet 
        self.updated_cone = False 

        #loop over every measurement 
        for k in range(np.shape(z)[0]):
            #z[k]: [x, y]

            pred_z = np.zeros((2, n_landmarks+1))
            pred_psi = np.zeros((n_landmarks+1, 2, 2))
            pred_H = np.zeros((n_landmarks+1, 2, 2 * (n_landmarks+1)+3))
            pi_k = np.zeros((n_landmarks+1, 1))

            # x,y,heading of landmark based on robot's current position 
            #create temporary new landmark at observed position 
            temp_mark = np.array([z[k][0],z[k][1]])


            # TODO: possibly fix axis
            state_mean_temp = np.append(state_mean_bar, temp_mark, axis=0)
            state_cov_temp = np.append(state_cov_bar, np.zeros((np.shape(state_cov_bar)[0], 2)), axis=1)
            state_cov_temp = np.append(state_cov_temp, np.zeros((2, np.shape(state_cov_bar)[1] + 2)), axis=0)
                   

            #initialize state covariance for new landmark proportional to range measurement squared
            for ii in range(np.shape(state_cov_temp)[0] - 2, np.shape(state_cov_temp)[0]):
                state_cov_temp[ii][ii] = (z[k][0]**2) / 130

            
            #index for landmark with maximum association 
            max_j = -1
            min_pi = 10 * np.ones((2,1))

            #loop over all landmarks and compute likelihood of correspondence with new landmark 
            for j in range(n_landmarks + 1):
                delta = np.array([state_mean_temp[2*j+3] - state_mean_temp[0],
                                  state_mean_temp[2*j+4] - state_mean_temp[1]])

                q = delta.dot(delta)
                r = np.sqrt(q)

                temp_theta = np.arctan2(delta[1], delta[0] - state_mean_temp[2])
                temp_theta = (temp_theta + 2 * np.pi) % (2 * np.pi)

                pred_z[:,j] = np.array([r, temp_theta], dtype=object)

                F_xj = np.zeros((5, 2 * (n_landmarks + 1) + 3))
                F_xj[0:3,0:3] = np.eye(3)
                F_xj[3:5,2*j+3:2*j+5] = np.eye(2)

                h_t = np.array([[-delta[0]/r, -delta[1]/r,  0,   delta[0]/r, delta[1]/r],
                                [delta[1]/q,  -delta[0]/q,   -1,  -delta[1]/q, delta[0]/q]])

                pred_H[j,:,:] = h_t @ F_xj
                pred_psi[j,:,:] = np.squeeze(pred_H[j,:,:]) @ state_cov_temp @ \
                                  np.transpose(np.squeeze(pred_H[j,:,:])) + self.Q_t

                if j < n_landmarks:
                    pi_k[j] = (np.transpose(z[k,0:2]-pred_z[:,j]) \
                                @ np.linalg.inv(np.squeeze(pred_psi[j,:,:]))) \
                                @ (z[k,0:2]-pred_z[:,j])
                else:
                    pi_k[j] = 0.84; # alpha: min mahalanobis distance to
                                    #        add landmark to map

                #tracking two best associations 
                if pi_k[j] < min_pi[0]:
                    min_pi[1] = min_pi[0]
                    max_j = j
                    min_pi[0] = pi_k[j]
        
            H = np.squeeze(pred_H[max_j,:,:])

            #best association must be significantly better than second better than second best 
            #otws, measurement is thrown out 
            if (min_pi[1] / min_pi[0] > 1.6):
                if max_j >= n_landmarks:
                    #new landmark is added, expand state and covariance matrices
                    state_mean_bar = state_mean_temp
                    state_cov_bar = state_cov_temp
                    n_landmarks += 1
                    
                    #indicate whether loop closure has occured if it hasn't already 
                    if not self.loop_closure: self.loop_closure = True 

                else:
                    #if measurement is associated with existing landmark, truncate h matrix to prevent dim. mismatch
                    H = H[:,0:2 * n_landmarks + 3]

                    K = state_cov_bar @ H.T @ np.linalg.inv(np.squeeze(pred_psi[max_j,:,:]))

                    state_mean_bar = state_mean_bar + K @ (z[k,0:2] - pred_z[:,max_j])

                    state_mean_bar[2] = (self.state_mean[2] + 2 * np.pi) % (2 * np.pi)

                    state_cov_bar = (np.eye(np.shape(state_cov_bar)[0]) - K @ H) @ state_cov_bar


                #seeing which cone (left, right) this observation is about
                if z[k][2] == 0 or (z[k][2] != 1 and z[k][0] <= 0): 
                    #assuming 0 is left OR if cone is orange and on left side (x-coor < 0) aka starting new lap  

                    #testing to see if landmark is updated 
                    if max_j != self.current_cone_left: #different indices 
                        self.current_cone_left = max_j
                        self.updated_cone = True 

                    #if left color, indicate cone is left OTWS indicate orange 
                    if z[k][2] == 0: self.current_cone_left_color = 0
                    else: self.current_cone_left_color = 2 

                elif z[k][2] == 1 or (z[k][2] != 0 and z[k][0] >= 0): 
                    #assuming 1 is right OR cone is orange and on right side (x-coor > 0) aka starting new lap 

                    if max_j != self.current_cone_right:
                        self.current_cone_right = max_j
                        self.updated_cone = True

                    # if right color, indicate cone is right OTWS indicate orange 
                    if z[k][2] == 1: self.current_cone_right_color = 1
                    else: self.current_cone_right_color = 2 
                else: 
                    print("something is going wrong lol since cone is not identifiable to be left or right")
                    return 
                
              
        #update state mean and covariance (map itself) 
        self.state_mean = state_mean_bar
        self.robot_state = self.state_mean[0:3]
        self.state_cov = state_cov_bar
        self.n_landmarks = n_landmarks
    

    def robot_cone_state(self):
        """
        Returns the robot pose and if current cone is updated/changed from previous cone observations, return 
        cone states 

        Transforms the cone w.r.t. robot from cone w.r.t. global and robot w.r.t. global 
        """ 
        if self.updated_cone: 
            # return pose of robot and pose of current landmarks if we changed the landmark our observations are associated with
            # return list of tuples [(x left cone, y left cone), (x right cone, y right cone), (x robot, y robot)]   
            
            T_wc_l = np.diag(np.asarray((1,1,1)))
            T_wc_l[0][2] = self.state_mean[2*self.current_cone_left+3] # x-coordinate of left cone 
            T_wc_l[1][2] = self.state_mean[2*self.current_cone_left+4] # y-coordinate of left cone 

            T_wc_r = np.diag(np.asarray((1,1,1)))
            T_wc_r[0][2] = self.state_mean[2*self.current_cone_right+3] # x-coordinate of right cone 
            T_wc_r[1][2] = self.state_mean[2*self.current_cone_right+4] # y-coordinate of right cone

            T_wr = np.diag(np.asarray((1,1,1)))
            T_wr[0][2] = self.robot_state[0]# x-coordinate of robot  
            T_wr[1][2] = self.robot_state[1] # y-coordinate of robot 

            T_rc_l = np.transpose(T_wr) @ T_wc_l 
            T_rc_r = np.transpose(T_wr) @ T_wc_r 

            V_rc_l = (T_rc_l[0][2], T_rc_l[1][2], self.current_cone_left_color)
            V_rc_r = (T_rc_r[0][2], T_rc_r[1][2], self.current_cone_right_color)

            V_wc_l = (self.state_mean[2*self.current_cone_left+3], self.state_mean[2*self.current_cone_left+4], self.current_cone_left_color)
            V_wc_r = (self.state_mean[2*self.current_cone_right+3], self.state_mean[2*self.current_cone_right+4], self.current_cone_right_color)

            # array of tuple of [robot state, change in robot state, left cone, right cone]
            return [(self.robot_state[0], self.robot_state[1], self.robot_state[2]),
                    (self.d_robot_state[0], self.d_robot_state[1], self.d_robot_state[2]), 
                    V_rc_l, V_rc_r]
        else:
            #array of tuple of [robot state, change in robot state]
            return [(self.robot_state[0], self.robot_state[1], self.robot_state[2]),
            (self.d_robot_state[0], self.d_robot_state[1], self.d_robot_state[2])]
            #return only robot state 

    def update_map(self, movement, measurements, new_time):
        """Updates map with given robot movement and landmark measurements after change in time dt 

        Parameters 
        ----------
        movement: list or numpy array 
            contains array of [new x coordinate, new y coordinate, new heading] 
            new positions of robot 
        measurement: list of lists or 2D numpy array 
            list of potential measurements of landmarks
            each potential landmark is represented as a list of [x, y, color] from the current robot position
        new_time: int 
            new time since previous robot movement and observation 

        Output
        ----------
        Executes prediction of robot position and update of landmark positions
        Updates the map of robot position and landmark postions 

        """
        self.__predict(movement, new_time)
        self.__update(measurements)

    def get_state(self, get_cov=False):
        if get_cov:
            return self.state_mean, self.state_cov
        else:
            return self.state_mean
    
    def get_loop_closure(self):
        """
        Returns whether the robot has completed a full loop around the track

        Output
        ----------
        True or False depending on loop closure status
        """
        return self.loop_closure

if __name__ == "__main__":
    map = Map()
    # print(map.get_state())
    move1 = np.array([1, 0, 0.4])
    #meas1 = np.array([[2, 0], [1, np.pi/2]])
    meas1 = np.array([[2, 0, 1], [1, np.pi/2, 1]])
    # map.update_map(move1, meas1, 1)
    print(map.get_state()) 
    print(map.updated_cone)
    print(map.robot_cone_state())
    #move2 = np.array([1, 0])
    #meas2 = np.array([[1.001, 0]])
    move2 = [1, 0, 0.4]
    meas2 = [[1.001, 0, 1]]
    map.update_map(move2, meas2, 1)
    if map.updated_cone == True: 
        print(map.robot_cone_state())

    print(map.get_state())
    print(map.updated_cone)
    print(map.robot_cone_state())