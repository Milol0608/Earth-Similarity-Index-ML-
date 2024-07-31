import openpyxl
import pandas as pd
import numpy as np

data = pd.read_excel(r'C:\Users\milob\OneDrive\Escritorio\PLANETS_WITH_ESI.xlsm')
data_array = np.array(data)
print(data_array)
print(data_array.shape)