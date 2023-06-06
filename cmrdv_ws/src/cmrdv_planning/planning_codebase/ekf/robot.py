# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:14:20 2018

@author: Attila
"""

import numpy as np

#definition of robot class
#containing motion and measurement noise covariances 
#used to generate the robot states and observations 
class Robot:
    def __init__(self, x_init, fov, Rt, Qt):
        x_init[2] = (x_init[2]+np.pi)%(2*np.pi)-np.pi
        self.x_true = x_init
        
        self.lo = np.empty((0,3))
        self.fov = np.deg2rad(fov)
        
        # noise covariances
        self.Rt = Rt #motion noise covar
        self.Qt = Qt #measurement noise covar 
    
    #motion model -- moving the robot by some change in position 
    def move(self,u):
        # u: list of changes in initial rotation, translation, and final rotation
        # returns new position of robot in x, y, theta  
        # Make noisy movement in environment

        # u = [v, w] => velocity, angular velocity
#        dt = 1
#        gamma = 0 # orientation error term
#        v = v # add error
#        w = w # add error
#        x[0] = x[0] - v/w*math.sin(x[2])+v/w*math.sin(x[2]+w*dt)
#        x[1] = x[1] + v/w*math.cos(x[2])-v/w*math.cos(x[2]+w*dt)
#        x[2] = x[2] + w*dt + gamma*dt  
        
        #motion_noise = np.matmul(np.random.randn(1,3),self.Rt)[0]
        [dtrans, drot1, drot2] = u[:3] #+ motion_noise #remove later
        #need to include some uncertainty / noise -- how to bsae this off of sensors? 
        
        x = self.x_true
        x_new = x[0] + dtrans*np.cos(x[2]+drot1)
        y_new = x[1] + dtrans*np.sin(x[2]+drot1)
        theta_new = (x[2] + drot1 + drot2 + np.pi) % (2*np.pi) - np.pi
        
        self.x_true = [x_new, y_new, theta_new]
        
        return self.x_true 
    
    # measurement model
    # given range and bearing of a landmark, calculate the x,y coordinates from current robot posiiton  
    def sense(self,lt):
        # lt: new observations
        # return landmark observations in x, y, theta based off current robot position 
        # Make noisy observation of subset of landmarks in field of view
        # *** noise also needs to be accounted for based on current sensors? 
        
        x = self.x_true
        observation = np.empty((0,3))
        
        fovL = (x[2]+self.fov/2+2*np.pi)%(2*np.pi)
        fovR = (x[2]-self.fov/2+2*np.pi)%(2*np.pi)
        
        for landmark in lt:
            rel_angle = np.arctan2((landmark[1]-x[1]),(landmark[0]-x[0]))
            rel_angle_2pi = (np.arctan2((landmark[1]-x[1]),(landmark[0]-x[0]))+2*np.pi)%(2*np.pi)
            # TODO: re-include and debug field of view constraints
            if (fovL - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi > 0 and (fovR - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi < 0:
                meas_range = np.sqrt(np.power(landmark[1]-x[1],2)+np.power(landmark[0]-x[0],2)) + self.Qt[0][0]*np.random.randn(1)
                meas_x = landmark[1]-x[1]
                meas_y = landmark[0]-x[0]
                # meas_range = np.sqrt(np.power(landmark[1]-x[1],2)+np.power(landmark[0]-x[0],2)) + self.Qt[0][0]*np.random.randn(1)
                # meas_bearing = (rel_angle - x[2] + self.Qt[1][1]*np.random.randn(1) + np.pi)%(2*np.pi)-np.pi
                observation = np.append(observation,[[meas_x[0], meas_y[0], landmark[2]]],axis=0)
                #observation = np.append(observation,[[meas_range[0], meas_bearing[0], landmark[2]]],axis=0): For range bearing measurments
                
        return observation
