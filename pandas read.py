import openpyxl
import pandas as pd
import numpy as np

data = pd.read_excel(r'C:\Users\File_Location) #Change to file location
data_array = np.array(data)
print(data_array)
print(data_array.shape)
