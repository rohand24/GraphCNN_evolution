import numpy as np
import json
import glob2
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import pdb

all_files = sorted(glob2.glob('./logs/Enzymes_1-23/'+'*.csv'), key=os.path.getmtime)

cycle = []
max_acc = []
avg_acc = []
model = []
for fname in all_files:

    acc = []
    curr_cycle = fname.split('/')[-1].split('.')[0]
    with open(fname, 'r+') as f:
        models = (json.load(f))
        for k,v in models.iteritems():
            k = k.encode('ascii')
            v = [i.encode('ascii') for i in v]
            acc.append(float(v[0]))
        
        max_acc.append(np.max(np.asarray(acc)))
        avg_acc.append(np.mean(np.asarray(acc)))
        cycle.append(all_files.index(fname))
    print('Done %s' %curr_cycle)
            
cycle = np.asarray(cycle)
max_acc = np.asarray(max_acc)
avg_acc = np.asarray(avg_acc)


plt.plot(cycle, max_acc,'r',label='Max Accuracy')
plt.plot(cycle, avg_acc, 'b--',label='Mean Accuracy')
plt.title('Fitness Change over Cycles')
plt.xlabel('Cycles')
plt.ylabel('Validation Accuracy')
plt.ylim(25,60)
plt.legend()
plt.grid()
plt.savefig("./plots/Max_avg_Enzymes_1-23_494cycle_output.png")


