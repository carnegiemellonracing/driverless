import math

def motion_model(t1, t2, state1, state2, left_cones, right_cones):
    dt = t2 - t1
    x1, y1, yaw1, _ = state1
    x2, y2, yaw2, _ = state2
    dx = x2 - x1
    dy = y2 - y1
    cmr_y = dx * math.cos(yaw1) + dy * math.sin(yaw1)
    cmr_x = dx * math.sin(yaw1) - dy * math.cos(yaw1)
    dyaw = yaw2 + 0.01 - yaw1
    
    left_cones2 = []
    right_cones2 = []
    for cone in left_cones:
        cone_x1 = cone[0] - cmr_x
        cone_y1 = cone[1] - cmr_y
        cone_y2 = -cone_x1 * math.sin(dyaw) + cone_y1 * math.cos(dyaw)
        cone_x2 = cone_x1 * math.cos(dyaw) + cone_y1 * math.sin(dyaw)
        left_cones2.append((cone_x2, cone_y2))
    for cone in right_cones:
        cone_x1 = cone[0] - cmr_x
        cone_y1 = cone[1] - cmr_y
        cone_y2 = -cone_x1 * math.sin(dyaw) + cone_y1 * math.cos(dyaw)
        cone_x2 = cone_x1 * math.cos(dyaw) + cone_y1 * math.sin(dyaw)
        right_cones2.append((cone_x2, cone_y2))

    return left_cones2, right_cones2

log_name = "logs/controls_test_node_1635660_1744979780276.log"
times = []
states = []
left_cones = []
right_cones = []

log = open(log_name, "r")

max_speed = 0
for line in log:
    if line[43:].startswith("Time:"):
        times.append(int(line[48:]))
    if line[43:].startswith("State:"):
        state = tuple(map(float,line[49:].split(",")))
        max_speed = max(max_speed, state[3])
        states.append(state)
    if line[43:].startswith("LeftCones:"):
        cones = list(map(lambda x : tuple(map(float, x.split(" "))),
                         line[53:].split(",")))
        left_cones.append(cones)
    if line[43:].startswith("RightCones:"):
        cones = list(map(lambda x : tuple(map(float, x.split(" "))),
                         line[54:].split(",")))
        right_cones.append(cones)

print(len(times))

for i in range(len(times)):
    print(i)
    old_ind = i
    new_ind = i + 3
    t1 = times[old_ind]
    t2 = times[new_ind]
    s1 = states[old_ind]
    s2 = states[new_ind]
    left_cones1 = left_cones[old_ind]
    left_cones2 = left_cones[new_ind]
    right_cones1 = right_cones[old_ind]
    right_cones2 = right_cones[new_ind]
    model_left_cones, model_right_cones = motion_model(t1, t2, s1, s2, left_cones1, right_cones1)
    for (cone_ground_truth, cone_modelled) in zip(left_cones2, model_left_cones):
        print(abs(cone_ground_truth[0] - cone_modelled[0]), abs(cone_ground_truth[1] - cone_modelled[1]))
        if (abs(cone_ground_truth[0] - cone_modelled[0]) > 0.001) or (abs(cone_ground_truth[1] - cone_modelled[1]) > 0.001):
            print(cone_ground_truth, cone_modelled, abs(cone_ground_truth[0] - cone_modelled[0]), abs(cone_ground_truth[1] - cone_modelled[1]))
    for (cone_ground_truth, cone_modelled) in zip(right_cones2, model_right_cones):
        print(abs(cone_ground_truth[0] - cone_modelled[0]), abs(cone_ground_truth[1] - cone_modelled[1]))
        if (abs(cone_ground_truth[0] - cone_modelled[0]) > 0.001) or (abs(cone_ground_truth[1] - cone_modelled[1]) > 0.001):
            print(cone_ground_truth, cone_modelled, abs(cone_ground_truth[0] - cone_modelled[0]), abs(cone_ground_truth[1] - cone_modelled[1]))