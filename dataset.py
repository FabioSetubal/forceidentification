# To save the dataset.csv file, uncomment line 69
# Importing libraries
import pandas as pd
from glob import glob # List directory files - https://pythonhelp.wordpress.com/2012/08/20/glob-listando-arquivos-de-diretorios/
from itertools import repeat # Create lists

# Reading files
files = sorted(glob('*.csv')) # this library reads files from a folder and creates a list with their names

# Defining variables
NPoints = 49
c_x = [-3, -2, -1, 0, 1, 2, 3]
c_y = [3, 2, 1, 0, -1, -2, -3]
P = []
dataset = pd.DataFrame()
x = pd.DataFrame()
y = pd.DataFrame()
point = pd.DataFrame()

# Handling of import data
for i in range(NPoints): # Force sweep at 49 points on the plate
    
    P.insert(i, 'P%d'%(10 + i)) # change feature column name every full scan of the sheet, start with P10 in Ansys files
    
    for cont in range(NPoints * i, NPoints * (i + 1)): # Sweep the features at 49 points on the sheet, for each force position
        
        if cont == NPoints * i: # Load the first file, delete the first lines and format the name of the feature column of point 1 on the sheet
            
            dataset_Temp = pd.read_csv(files[cont], encoding = 'UTF-8', sep = ',', header = 6) # load file and delete first 5 lines, disregard blank line
            dataset_Temp = dataset_Temp.rename(columns = {'P8':'Amplitude', 'P9':'Frequency', P[i]:'F%d'%(cont + 1 - NPoints * i)}) # rename column from P10 to F1
              
        else:
            
            new_columns = pd.read_csv(files[cont], encoding = 'UTF-8', sep = ',', header = 6, usecols = [2]) # load only feature column in following files - usecols = [2], delete first 5 lines, disregard blank lines
            new_columns = new_columns.rename(columns = {P[i]:'F%d'%(cont + 1 - NPoints * i)}) # Rename column from P11 onwards to F2 onwards
            dataset_Temp.insert(loc = cont + 2 - NPoints * i, column = 'F%d'%(cont + 1 - NPoints * i), value = new_columns) # insert the feature column in the dataset file and update it up to 49 features - F49
            
    dataset = dataset.append(dataset_Temp, ignore_index = True) # Repeat the process and put the information below in the dataset and update it

# Create column of point numbering
for cont in range(NPoints):
    
    point_Temp = [(cont + 1) for i in repeat(None, 2500 * 1)]
    point_Temp = pd.DataFrame(point_Temp, columns = ['Point'])
    point = point.append(point_Temp, ignore_index = True)

# Create column X
for cont in range(7):

    x_Temp = [c_x[cont] for i in repeat(None, 2500 * 7)]
    x_Temp = pd.DataFrame(x_Temp, columns = ['X'])
    x = x.append(x_Temp, ignore_index = True)

# Create column Y
for i in range(7):

    for cont in range(7):
    
        y_Temp = [c_y[cont] for i in repeat(None, 2500 * 1)]
        y_Temp = pd.DataFrame(y_Temp, columns = ['Y'])
        y = y.append(y_Temp, ignore_index = True)        

# Insert the Point, X and Y columns into the dataset file
dataset.insert(loc = 0, column = 'Point', value = point)
dataset.insert(loc = 2, column = 'X', value = x)
dataset.insert(loc = 3, column = 'Y', value = y)

# Generating .csv file
#dataset.to_csv(r'dataset.csv', sep = ',', index = None, encoding='utf-8', header='true')
