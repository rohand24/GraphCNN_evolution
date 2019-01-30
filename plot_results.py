import numpy as np
import json
import glob2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import pdb
import os

all_files = sorted(glob2.glob('./logs/Enzymes_1-23/'+'*.csv'), key=os.path.getmtime)

cycle = []
acc = []
model = []
for fname in all_files:

    curr_cycle = fname.split('/')[-1].split('.')[0]
    with open(fname, 'r+') as f:
        models = (json.load(f))
        for k,v in models.iteritems():
            k = k.encode('ascii')
            v = [i.encode('ascii') for i in v]
            acc.append(float(v[0]))
            model.append(k)
            cycle.append(all_files.index(fname))
    print('Done %s' %curr_cycle)
            
cycle = np.asarray(cycle)
acc = np.asarray(acc)
model = np.asarray(model)

dataset = pd.DataFrame({'cycle':cycle,'accuracy':acc,'model':model})

dataset.sort_values(by=['cycle'], ascending = False)
ax = sns.relplot(x="cycle", y="accuracy", hue="model", dashes=False, markers=['o', 'v', 's'], kind="line", data=dataset,palette=("bright"))
ax.savefig("./plots/Enzymes_Enzymes_1-23_494cycle_output.png")


