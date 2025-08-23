### Setup
This is a way to run the controls workflow locally, assuming you have a GPU.

This docker image containerizes the repo and allows GPU passthrough.

Unfortunately, I was not able to get CAN and DRIVERLESS to be copied into the image through the context, so the only requirement (apart from the GPU) is that **can** and driverless_ws named **tmp_driverless_ws** is within this same folder.

###