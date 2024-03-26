import matplotlib.pyplot as plt, json
import numpy as np

plt.ion()
    
while True:
    try:
        best_overtime = np.array(json.load(open('generations.json', 'r+')))
    except Exception as e:
        continue
    
    plt.plot(np.arange(len(best_overtime)) + 1, best_overtime)
    plt.draw()
    plt.pause(0.1)
    plt.clf()