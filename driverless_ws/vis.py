import matplotlib.pyplot as plt
import re

input = """Landmark 0: (-4.594687, 5.430464)         
Landmark 1: (1.967950, 4.239662)        
Landmark 2: (1.538826, 7.457808)      
Landmark 3: (-2.009286, 4.823530)      
Landmark 4: (-2.100775, 9.195071)         
Landmark 5: (2.734748, 2.223263)          
Landmark 6: (0.807712, 6.122287)          
Landmark 7: (-3.377982, 7.109918)                     
Landmark 7: (-3.377982, 7.109918)      
Landmark 8: (-1.733143, 2.628232)      
Landmark 9: (-6.942954, 8.613847)       
Landmark 11: (3.021747, 10.243058)     
Landmark 12: (3.422997, 16.249179)                                                                    ├───────────────────────────────────────────────────
Landmark 13: (3.787509, 18.715111)     
Landmark 14: (4.611504, 20.834904)     
Landmark 15: (4.637874, 5.291833)        
Landmark 16: (7.567601, 12.024079)      
Landmark 17: (10.199362, 14.983701)     
Landmark 18: (2.994543, -1.285342)       
Landmark 19: (7.635023, 4.431190)         
Landmark 20: (5.260285, 2.464374)         
Landmark 21: (15.104269, 13.125831)       
Landmark 22: (8.460316, 5.769965)         
Landmark 23: (4.444549, 0.574193)         
Landmark 24: (11.669410, 3.583154)        
Landmark 25: (13.911226, 9.267782)        
Landmark 26: (14.627269, 5.707978)        
Landmark 27: (18.918588, 3.273233)        
Landmark 28: (7.119548, 0.042108)      
Landmark 29: (13.514128, 3.794054)"""

landmarks = input.split("\n")
x = []
y = []
for i in range(len(landmarks)):
    tokens = re.split(r'[( ,)]', landmarks[i].strip())
    x.append(float(tokens[3]))
    y.append(float(tokens[5]))

print(x)
print(y)
plt.scatter(x, y)
plt.show()
# print(landmarks)