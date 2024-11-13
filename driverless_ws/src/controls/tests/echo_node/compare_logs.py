"""
[INFO] [1725133795.820627455] [controller]: mppi step complete. info:
Action:
  swangle (rad): 0.191232
  torque_fl (Nm): 3.63583
  torque_fr (Nm): 3.63583
  torque_rl (Nm): 0
  torque_rr (Nm): 0
Projected State:
  x (m): 0.27797
  y (m): 0.0293018
  yaw (rad): 0.0333356
  speed (m/s): 4.07983
MPPI Step Latency (ms): 2
Total Latency (ms): 77
"""
import re
import argparse

class Action:
    def __init__(self, swangle, torque_fl, torque_fr):
        self.swangle = swangle
        self.torque_fl = torque_fl
        self.torque_fr = torque_fr

    def __repr__(self):
        return f"Action(swangle={self.swangle}, torque_fl={self.torque_fl}, torque_fr={self.torque_fr})"

class State:
    def __init__(self, x, y, yaw, speed):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = speed

    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, yaw={self.yaw}, speed={self.speed})"

def parse_file(file_path, max_length):
    action_state_pairs = []

    # Regular expressions for extracting data
    action_pattern = re.compile(r'Action:\n\s*swangle \(rad\): ([\d.]+)\n\s*torque_fl \(Nm\): ([\d.]+)\n\s*torque_fr \(Nm\): ([\d.]+)\n\s*torque_rl \(Nm\): [\d.]+\n\s*torque_rr \(Nm\): [\d.]+')
    state_pattern = re.compile(r'Projected State:\n\s*x \(m\): (-?[\d.]+(?:e-?\d+)?)\n\s*y \(m\): (-?[\d.]+(?:e-?\d+)?)\n\s*yaw \(rad\): (-?[\d.]+(?:e-?\d+)?)\n\s*speed \(m/s\): (-?[\d.]+(?:e-?\d+)?)')
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # for state_match in state_pattern.finditer(content):
        #     # swangle, torque_fl, torque_fr = map(float, action_match.groups())
            
        #     # Find the corresponding State
        #     x, y, yaw, speed = map(float, state_match.groups())
        #     state = State(x, y, yaw, speed)
        #     action_state_pairs.append((Action(0,0,0), state))
            
        #     if len(action_state_pairs) >= max_length:
        #         break

        # Find all matches for Action
        for action_match in action_pattern.finditer(content):
            swangle, torque_fl, torque_fr = map(float, action_match.groups())
            
            # Find the corresponding State
            state_match = state_pattern.search(content, action_match.end())
            if state_match:
                x, y, yaw, speed = map(float, state_match.groups())
                action = Action(swangle, torque_fl, torque_fr)
                state = State(x, y, yaw, speed)
                action_state_pairs.append((action, state))
            
            if len(action_state_pairs) >= max_length:
                break
    
    return action_state_pairs

def find_difference(asp1, asp2):
    compare_length = min(len(asp1), len(asp2))
    allowable_diff = 0.4
    swangle_range = 2 * 0.33
    torque_range = 2 * 4

    for i in range(compare_length):
        action1, state1 = asp1[i]
        action2, state2 = asp2[i]

        if abs(action1.swangle - action2.swangle) > allowable_diff * swangle_range:
            print(f"Action {i+1}: Action1: {action1}, Action2: {action2}")
            return

        if abs(action1.torque_fl - action2.torque_fl) > allowable_diff * torque_range:
            print(f"Action {i+1}: Action1: {action1}, Action2: {action2}")
            return   

        if abs(action1.torque_fr - action2.torque_fr) > allowable_diff * torque_range:
            print(f"Action {i+1}: Action1: {action1}, Action2: {action2}")
            return

        # if abs(state1.x - state2.x) > allowable_diff * (state1.x + state2.x) / 2:
        #     print(f"State {i+1}: State1: {state1}, State2: {state2}")
        #     return

        # if abs(state1.y - state2.y) > allowable_diff * (state1.y + state2.y) / 2:
        #     print(f"State {i+1}: State1: {state1}, State2: {state2}")
        #     return

        # if abs(state1.yaw - state2.yaw) > allowable_diff * (state1.yaw + state2.yaw) / 2:
        #     print(f"State {i+1}: State1: {state1}, State2: {state2}")
        #     return

        # if abs(state1.speed - state2.speed) > allowable_diff * (state1.speed + state2.speed) / 2:
        #     print(f"State {i+1}: State1: {state1}, State2: {state2}")
        #     return

    print("Refactor successful! No diffs found")


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Parse files containing Action and State information.")
    parser.add_argument('file_path1', type=str, help="Path to the first file")
    parser.add_argument('file_path2', type=str, help="Path to the second file")
    args = parser.parse_args()

    max_length = 1000

    # Parse both files
    print(f"Parsing file: {args.file_path1}")
    action_state_pairs1 = parse_file(args.file_path1, max_length)
    print(action_state_pairs1[:5])
    print(f"Parsing file: {args.file_path2}")
    action_state_pairs2 = parse_file(args.file_path2, max_length)

    find_difference(action_state_pairs1, action_state_pairs2)

if __name__ == '__main__':
    main()


