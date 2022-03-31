import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../data/external/train.csv')

subtypes = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

# Removes all columns except those listed in subtypes
subtype = df[subtypes]

# Creates a new dataframe from the counts of each column
subtype = subtype[subtype > 0].count()
    
subtype.plot(kind='bar', title='Subtype Attribute Count')

plt.savefig('../../reports/figures/subtype.png', bbox_inches = "tight")
