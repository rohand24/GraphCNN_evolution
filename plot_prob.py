import numpy as np
import json
import glob2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import pdb


def main():

    prob_data = []
    log_path = './logs/Enzymes_1-23'

    with open(log_path +'/mutation_log.txt', 'r+') as f:
        all_lines = f.readlines()
        all_probs = all_lines[1::2]
        for line in all_probs:
            probs = [float(i) for i in line.strip().split('\t')]
            prob_data.append(np.asarray(probs))
            
    #pdb.set_trace()
    cycles = len(prob_data)
    mutations = ['Learning_rate','Add_fc','Remove_fc','Add_conv','Remove_conv','Add_skip','Remove_skip','Add_EdgeConv','Remove_EdgeConv', 'Add_one_to_one','Remove_one_to_one', 'Add_attention','Remove_attention', 'Regularization_change','Replace_filter', 'Pool_mutation', 'Pool-gep_mutation']

    prob_df = pd.DataFrame(data = prob_data, columns = mutations)
    prob_df['Cycles'] = range(cycles) 
    df1 = prob_df[['Learning_rate', 'Regularization_change','Cycles']]
    df2 = prob_df[['Add_fc','Add_skip', 'Add_attention','Cycles']]
    df3 = prob_df[['Add_conv','Add_one_to_one', 'Add_EdgeConv','Cycles']]
    df4 = prob_df[['Remove_conv','Remove_one_to_one', 'Remove_EdgeConv','Cycles']]
    df5 = prob_df[['Remove_fc','Remove_skip','Remove_attention','Cycles']]
    df6 = prob_df[['Pool_mutation','Pool-gep_mutation', 'Replace_filter', 'Cycles']]
    sns.set_style("ticks")
    sns.set_context("paper")
    df_list = [df1,df2,df3, df4, df5, df6]
    for i in range(len(df_list)):
        ax = sns.relplot(x="Cycles", y="value", hue='variable', dashes=False, markers=['o'], kind="line", data=df_list[i].melt('Cycles'),palette=("bright"))
        ax.set(xlabel='Cycles', ylabel='Probability',title = 'Mutation Probability vs Cycles')
        
        ax.savefig("./plots/Enzymes_1-23_494cycle_prob_plot"+str(i)+".png")
    
    

if __name__ == "__main__":
    main()
      
