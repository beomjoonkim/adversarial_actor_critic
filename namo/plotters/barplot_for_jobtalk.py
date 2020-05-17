from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(2)
money = ['Guided planner', 'Unguided planner']


fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(formatter)
plt.bar(x, [1550, 3500])
plt.xticks(x, ('Guided \nplanner', 'State-of-the-art \nplanner'), fontsize=14)
plt.ylabel('Planning times in seconds', fontsize=14)
import pdb;pdb.set_trace()
