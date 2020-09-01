import numpy as np
import pandas as pd

a = pd.DataFrame({'ID':[0,1,2,1], 'Frame':0, 'xmin':0, 'ymin':0, 'xmax':0, 'ymax':0, 'Conf':0, 'Label':'test', 'Occlude':0})
b = a.loc[:,'ID']
a.append(a)
print(list(set(a['ID'].values.tolist())))

# boxes = [[[0, 3, 0, 3, 0.9237443, 0.9237443, 1], 
#           [1, 4, 1, 4, 0.9179137, 0.9179137, 1], 
#           [3, 5, 3, 5, 0.9790607, 0.9790602, 1]]]

# for box in boxes[0]:
#     print(box)