import os
import matplotlib.pyplot as plt

f = open("history.txt")

lines = []
for line in f:
    lines.append(line)

history = 0
prediction = 0
for line in lines:
    print(line)
    if line == "---- BEGIN HISTORY ---\n":
        history += 1
    if line == "---- BEGIN PREDICTION ----\n":
        prediction += 1

cnt = min(history, prediction)

def parse_entries(lines):
    results = []
    
    for line in lines:
        if "Time:" not in line:
            continue

        entry = {}
        line = ''.join(line.split())
        parts = line.split("|||")
        for part in parts:
            item = part.split(":")
            print(item)
            if len(item) == 2:
                if item[0] == "Time":
                    item[1] = item[1].removesuffix("ns")
                entry[item[0]] = item[1]
        
        # Only include entries with time
        if "Time" in entry:
            results.append(entry)
    
    return results

def is_pred_dict(dict):
    for key in dict.keys():
        if "Pred" in key:
            return True
        
    return False

def create_time_stamp_pairs(dicts):
    time_dict = {}
    for dict in dicts:
        curr_time = dict["Time"]
        curr_pair = time_dict.get(curr_time, [])
        del dict["Time"]
        if is_pred_dict(dict):
            curr_pair.append(("Pred", dict))
        else:
            curr_pair.append(("True", dict))

        time_dict[curr_time] = curr_pair

    return time_dict

def pretty_print(D):
    for key in D.keys():
        print(D[key])

def filter_dicts(Dicts):
    new_Dict = {}
    for key in Dicts.keys():
        if len(Dicts[key]) != 2:
            continue
        else:
            new_Dict[key] = Dicts[key]

    return new_Dict

def get_translation_map():
    translations_one_way = {
        'PredictedSpeed' : 'Speed',
        'PredictedX' : 'X',
        'PredictedY' : 'Y',
        'PredictedYaw' : 'Yaw',
    }

    translations = {}

    for key, val in translations_one_way.items():
        translations[val] = key
        translations[key] = val

    return translations

dicts = parse_entries(lines)
T = get_translation_map()

time_stamp_pairs = create_time_stamp_pairs(dicts)   
filtered_time_stamp_pairs = filter_dicts(time_stamp_pairs)

L = list(filtered_time_stamp_pairs.items())

L = sorted(L, key = lambda x : x[0])

for i in L: 
    print(i)

key_count = {}
for time, label_dicts_pairs in L:
    label1, dict1 = label_dicts_pairs[0]
    label2, dict2 = label_dicts_pairs[1]

    for key in dict1.keys():
        if key in T and T[key] in dict2.keys():
            key_count[key] = key_count.get(key, 0) + 1

valid_keys = []
error = 20
for key in key_count.keys():
    # print(key_count[key])
    if abs(key_count[key] - len(L)) <= error:
        valid_keys.append(key)

super_data = {}
for key in valid_keys:
    super_data[key] = []
# print(super_data.keys())
for instance in L:
    time, dictlist = instance
    label1, dict1 = dictlist[0]
    label2, dict2 = dictlist[1]
    for key in dict1.keys():
        if key in valid_keys:
            if T[key] in dict2.keys():
                super_data[key].append((dict1[key], dict2[T[key]]))

for key in super_data.keys():
    path = f"endpoints/{key}.txt"
    f = open(path, "w")    
    for point in super_data[key]:
        (x, y) = point
        point = (float(x), float(y))
        f.write(str(point))
        f.write("\n")

    f.close()

def plot_endpoints(directory="endpoints"):
    """
    Reads all .txt files in the specified directory, parses tuples of the form (x1, x2),
    and creates side-by-side subplots with the plot title "Time vs. {Property}".
    
    Parameters:
    - directory (str): The folder containing the endpoint files.
    """
    # Get all .txt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    n_files = len(files)
    
    if n_files == 0:
        print(f"No .txt files found in the directory: {directory}")
        return
    
    # Create subplots horizontally
    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 4))
    
    # In case there's only one file, convert axes to a list for uniform processing.
    if n_files == 1:
        axes = [axes]
    
    # Iterate through each file and plot its data
    for ax, file in zip(axes, files):
        # Extract property name from file name (without extension)
        property_name = os.path.splitext(file)[0]
        x1_values = []
        x2_values = []
        file_path = os.path.join(directory, file)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                # Remove enclosing parentheses if they exist
                if line.startswith('(') and line.endswith(')'):
                    line = line[1:-1]
                
                # Split the line based on comma
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        # Convert both parts to float after stripping whitespace
                        x1 = float(parts[0].strip())
                        x2 = float(parts[1].strip())
                    except ValueError:
                        # Skip lines that can't be parsed into two floats
                        continue
                    x1_values.append(x1)
                    x2_values.append(x2)
        
        # Plot the parsed data into the current subplot
        ax.plot(x1_values, x2_values, marker='o', linestyle='-')
        ax.set_xlabel("Time (unit)")
        ax.set_ylabel(property_name)
        ax.set_title(f"Time vs. {property_name}")
    
    plt.tight_layout()
    plt.show()

print(super_data)
# To run the function, simply call:
plot_endpoints()


