# cmr-cmrdv

## Development steps
*This may need to be completed in a Docker container* \
In order to run any module developed on a subteam must first install all the required packages defined in `requirements.txt`

```
# python 2
pip install -r requirements.txt

# python3 
pip3 install -r requirements.txt
```

Then in `~/.bashrc` or `~/.zshrc` add the below line to add this repository to your `PYTHONPATH`
```
export PYTHONPATH="{path/to/cmrdv/cmrdv_ws/src/}:$PYTHONPATH"
```
To find the path to repository: 
```
pwd
```

Then source your `~/.bashrc` or `~/.zshrc` file to let changes take hold. Adding to this file ensures each terminal you open has already added the correct path. 
```
source ~/.bashrc
```

When testing a file, ensure you have written a main function in your file: `if __name__ == "__main__":`, then run your module as normal.

## Compile Workspace

To build all packages:
```
colcon build 
```
To build only a specific package: 
```
colcon build --packages-select <package-name>
```

To run a node: 
```
ros2 run <package-name> <entrypoint>
```
- package name is the package that the node belongs to
- entrypoint is defined in `setup.py` for each package
