# from operator import truediv
# import gtsam
# import matplotlib.pyplot as plt
# import numpy as np
# import math

# class GraphSlam: 
#     def __init__(self, lc_limit):
#         # Intialize factor graph, values, and noise model
#         # Noise model is predefined here
#         self.factor_graph = gtsam.NonlinearFactorGraph()
#         self.vals = gtsam.Values()
#         self.noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.01])

#         self.robot_poses = []
#         self.cone_transforms = []
#         self.global_cones = []
#         self.lc_limit = lc_limit
#         self.lc_curr = 0
#         self.lc_collect = False
#         self.optimized = False
#         self.lap = 0 
#         self.lap_poses = 0
#         self.new_cone_pair = None
#         self.new_pose = None
#         self.vehicle_state = None
#         self.cone_positions = None

#     def __sees_orange_cones(self):
#         # Cone Pair consists of a 2d list. [ [cone_pose_blue, color], [cone_pose_yellow, color] ]
#         # Checking if both cones have color = 2 (orange) -- Should this be both (or just either or) ??
#         if self.new_cone_pair[0][1] == 2 and self.new_cone_pair[1][1] == 2: return True
#         else: return False

#     def __get_diff_pose(self, pose_a, pose_b):
#         diff_x = pose_b.x - pose_a.x
#         diff_y = pose_b.y - pose_a.y
#         diff_theta = pose_b.theta - pose_a.theta
#         return gtsam.Pose2(diff_x, diff_y, diff_theta)

#     def __add_prior_factor(self):
#         self.factor_graph.add(gtsam.PriorFactorPose2(0, self.new_pose, self.noise_model))

#     def __add_odom(self):
#         odom = self.__get_diff_pose(self.robot_poses[-1], self.new_pose)
#         self.factor_graph.add(gtsam.BetweenFactorPose2(self.lap_poses - 1, self.lap_poses, odom, self.noise_model))

#     def __add_val(self):
#         self.vals.insert(self.lap_poses, self.new_pose)

#     def __collect(self):
#         if len(self.robot_poses) == 0:
#             self.factor_graph = self.__add_prior_factor()
#         else: 
#             self.factor_graph = self.__add_odom()

#         self.robot_poses.append(self.new_pose)

#         # Only collect cone poses for the first lap
#         # First lap will be most accurate
#         if self.lap == 1:
#             self.cone_transforms.append(self.new_cone_pair)

#         self.vals = self.__add_val(self.new_pose, self.lap_poses)

#     def __add_lc(self):

#         # Determining indices for poses in loop closure constraints
#         i = self.lc_curr
#         j = self.lap_poses * (self.lap - 1) + self.lc_curr

#         pose_old = self.robot_poses[i]

#         # Retrieve left cone pose for old robot pose
#         cone_left_i = pose_old.compose(self.cone_transforms[i][0][0])

#         # Transformation from left cone pose (new robot pose) to new robot pose
#         pose_new_transform = self.new_cone_pair[0][0].inverse()

#         # Apply transformation to left cone pose (old robot pose)
#         # Gets new robot pose with respect to old robot pose
#         pose_new = cone_left_i.compose(pose_new_transform)
        
#         # Get difference between old and new robot pose localized in same frame
#         pose_diff = self.__get_diff_pose(pose_old, pose_new)

#         # Add loop closure constraint (same noise model as odometry constraints)
#         self.factor_graph.add(gtsam.BetweenFactorPose2(i, j, pose_diff, self.noise_model))

#     def __optimize(self):
#         optimizer = gtsam.GaussNewtonOptimizer(self.factor_graph, self.vals)
#         self.vals = optimizer.optimize()

#     def __vals_to_poses(self):
#         poses = []
#         for k in self.vals.keys():
#             p = self.vals.atPose2(k)
#             poses.append(p)
#         return poses

#     def __average_poses(pose_a, pose_b):
#         avg_x = (pose_b.x + pose_a.x) / 2
#         avg_y = (pose_b.y + pose_a.y) / 2
#         avg_theta = (pose_b.theta + pose_a.theta) / 2
#         return gtsam.Pose2(avg_x, avg_y, avg_theta)

#     def __smooth_cones(self, new_cones):
#         smooth_cones = []
#         # Number of old cones and new cones should always be the same if algo works correctly
#         assert(len(self.global_cones) == len(new_cones))

#         for i in range(len(self.global_cones)):

#             # Get average global pose for left cone
#             smooth_left_pose = self.__average_poses(self.global_cones[i][0][0], new_cones[i][0][0])

#             # Get average global pose for right cone
#             smooth_right_pose = self.__average_poses(self.global_cones[i][1][0], new_cones[i][1][0])

#             # Creating cone pair for average globalized poses, adding assigned color to array
#             smooth_left = [smooth_left_pose, self.global_cones[i][0][1]]
#             smooth_right = [smooth_right_pose, self.global_cones[i][1][1]]
#             smooth_cones.append([smooth_left, smooth_right])

#         return smooth_cones

#     def __optimize_cones(self):
#         new_cones = []
#         for i in range(len(self.cone_transforms)):

#             # Get global pose for left cone
#             cone_left_pose = self.robot_poses[i].compose(self.cone_transforms[i][0][0])

#             # Get global pose for right cone
#             cone_right_pose = self.robot_poses[i].compose(self.cone_transforms[i][1][0])

#             # Creating cone pair for globalized poses, adding assigned color to array
#             cone_left = [cone_left_pose, self.cone_transforms[i][0][1]]
#             cone_right = [cone_right_pose, self.cone_transforms[i][1][1]]

#             new_cones.append([cone_left, cone_right])
        
#         # If first time optimizing, global cones are the new cones
#         if len(self.global_cones) == 0:
#             self.global_cones = new_cones
#         # Otherwise, average old global cones with new cones to produce more accurate global cones (smoothing effect)
#         else:
#             self.global_cones = self.__smooth_cones(new_cones)

#         return self.global_cones

#     def __pose_from_ekf(self, ekf_output):
#         x = ekf_output[0][0]
#         y = ekf_output[0][1]
#         theta = ekf_output[0][2]
#         self.new_pose = gtsam.Pose2(x, y, theta)

#     def __cone_pair_from_ekf(self, ekf_output):
#         c_l_x = ekf_output[2][0]
#         c_l_y = ekf_output[2][1]
#         c_l_theta = math.atan2(c_l_y, c_l_x)
#         c_l_color = ekf_output[2][2]

#         c_r_x = ekf_output[3][0]
#         c_r_y = ekf_output[3][1]
#         c_r_theta = math.atan2(c_r_y, c_r_x)
#         c_r_color = ekf_output[3][2]

#         cone_left = gtsam.Pose2(c_l_x, c_l_y, c_l_theta)
#         cone_right = gtsam.Pose2(c_r_x, c_r_y, c_r_theta)
#         self.new_cone_pair = [ [cone_left, c_l_color], [cone_right, c_r_color] ]

#     def __parse_ros_cones(self):
#         cones = []
#         # Iterating over every cone pair
#         for c_p in self.global_cones:
#             # Iterating over every cone in each pair
#             for c in c_p:
#                 # Getting cone pose x and y
#                 x = c[0].x
#                 y = c[0].y
#                 cones.append(x)
#                 cones.append(y)
#         self.cone_positions = cones
    
#     def __parse_ros_vehicle_state(self, ekf_output):
#         state = [0] * 6
#         most_recent_pose = self.robot_poses[-1]
#         state[0] = most_recent_pose.x # x
#         state[1] = most_recent_pose.y # y
#         state[2] = most_recent_pose.theta # theta
#         state[3] = ekf_output[1][0] # dx
#         state[4] = ekf_output[1][1] # dy
#         state[5] = ekf_output[1][2] # dtheta
#         self.vehicle_state = state

#     def run_graph_slam(self, ekf_output):
    
#         self.new_pose = self.__pose_from_ekf(ekf_output)
#         self.new_cone_pair = self.__cone_pair_from_ekf(ekf_output)
#         self.__collect()

#         if self.__sees_orange_cones():
#             # Lap number starts as 0
#             # Gone around track at least once, start collecting loop closures
#             if self.lap > 1:
#                 self.factor_graph = self.__add_lc()
#                 # Collecting loop closures now
#                 self.lc_collect = True
#                 self.lc_curr += 1
#             else:
#                 # Lap 1 begins
#                 self.lap = 1

#         # If loop closure collection started, add loop closure
#         elif self.lc_collect == True:
#             self.factor_graph = self.__add_lc()
#             self.lc_curr += 1
        
#         # If desired loop closures collected, begin graph optimization
#         if self.lc_curr == self.lc_limit:
#             self.vals = self.__optimize()
#             self.robot_poses = self.__vals_to_poses()

#             # Get global cone poses
#             self.global_cones = self.__optimize_cones()

#             # Turn off loop closure collection, reset increment
#             self.lc_collect = False
#             self.lc_curr = 0

#             # Letting slam node know graph has been optimized
#             self.optimized = True
        
#         # Increment lap pose
#         self.lap_poses += 1

#         # Parse vehicle state and cone positions to return
#         self.__parse_ros_cones()
#         self.__parse_ros_vehicle_state(ekf_output)

#         return self.vehicle_state, self.cone_positions, self.optimized
