import numpy as np  
import pandas as pd  
  
txt = np.loadtxt('dataset2.txt')  
txtDF = pd.DataFrame(txt)  
txtDF.to_csv('dataset.csv',index=False)  
