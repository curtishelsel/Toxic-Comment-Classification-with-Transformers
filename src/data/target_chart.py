import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../data/external/train.csv')

# Removes all columns except target
target = df[['target']]

# Removes all values less than .5 but greater than 0
target = target[(target['target'] == 0) | (target['target'] >= 0.5)]

# Replaces all values greater than or equal to 0.5 to 1
target[target >= .5] = 1

# Creases a new dataframe from the counts of each value either 0 or 1
target = target['target'].value_counts()

target.plot(kind='bar', title='Target Imbalance',
                xlabel='Target Value', ylabel='Count')

# Changes from scientific notation to normal
plt.ticklabel_format(useOffset=False, style='plain', axis='y')

plt.savefig('../../reports/figures/targetimbalance.png', bbox_inches = "tight")
