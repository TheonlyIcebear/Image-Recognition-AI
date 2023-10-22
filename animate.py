<<<<<<< HEAD
<<<<<<< HEAD
import matplotlib.pyplot as plt, json
import numpy as np

plt.ion()
    
while True:
    try:
        best_overtime = np.array(json.load(open('generations.json', 'r+')))
    except:
        continue
    
    plt.plot(np.arange(len(best_overtime)) + 1, best_overtime)
    plt.draw()
    plt.pause(0.0001)
=======
import matplotlib.pyplot as plt, json
import numpy as np

plt.ion()
    
while True:
    try:
        best_overtime = np.array(json.load(open('generations.json', 'r+')))
    except:
        continue
    
    plt.plot(np.arange(len(best_overtime)) + 1, best_overtime)
    plt.draw()
    plt.pause(0.0001)
>>>>>>> 72debf778cab3d43425f2606e0f1fdbac5082207
=======
import matplotlib.pyplot as plt, json
import numpy as np

plt.ion()
    
while True:
    try:
        best_overtime = np.array(json.load(open('generations.json', 'r+')))
    except:
        continue
    
    plt.plot(np.arange(len(best_overtime)) + 1, best_overtime)
    plt.draw()
    plt.pause(0.0001)
>>>>>>> 72debf778cab3d43425f2606e0f1fdbac5082207
    plt.clf()