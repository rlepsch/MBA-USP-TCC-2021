# Libs
import pandas as pd
import numpy as np
import datetime as dt
import time

#------------------------------------------------------

# file loading
file_name = 'Internacoes.xlsx' 
df = pd.read_excel(file_name, index_col=0)
#df = pd.read_excel('Internacoes.xlsx', engine = 'openpyxl')

# Create a logical field to check whether or not a row is taken from original data
df['Original'] = 1

# Create a key ('Municipio' + 'Nome do Hospital' field) and a hospital list
df['key'] = df['Municipio'] + ' - ' + df['Nome Fantasia']
hospital_list = df['key'].unique().tolist()

# Define the full period of time (from 2020-12-03 to today)
d_first = dt.date(2020, 12, 3)
d_today = dt.date.today()
#d_today = d_today - dt.timedelta(days=1) #yesterday
idxs_period = pd.date_range(d_first, d_today, freq = "D")
#idxs_period = idxs_period.astype('datetime64[ns]')

# Select all the samples from each hospital
hosp_ind = ['Pacientes Enfermaria','Pacientes UTI']
df_final = []
for i in hospital_list:

    # Case 0: 'Pacientes Enfermaria'
    #----------------------------------
    df_hosp_Enf = df.loc[(df['key'] == i) & (df['Nomes de medida'] == hosp_ind[0])]    

    # Set the 'Data' field as index, and reindex in order to add the missing rows from 'idxs_period'
    df_hosp_Enf.set_index('Data', inplace=True)
    df_hosp_Enf.index = pd.DatetimeIndex(df_hosp_Enf.index)
    df_hosp_Enf = df_hosp_Enf.reindex(idxs_period) #expand the rows to get 'idxs_periods'
 
    # Fill NaN from 'Original' field with 0.0 (i.e., set the row as an artificially created one)
    df_hosp_Enf['Original'] = df_hosp_Enf['Original'].fillna(0)

    # Fill the NaN values of the rows after the original rows, and then, fill the remaining rows 
    # before the first filled row
    df_hosp_Enf = df_hosp_Enf.ffill().bfill()

    # Heuristic 1: set zero to the artificially created rows (Original=0) whose "Valores de medida = 1"
    # (because the current row is a ghost one generated from an original sample with value = 1) 
    mask = (df_hosp_Enf['Original'] == 0) & (df_hosp_Enf['Valores de medida'] == 1)
    df_hosp_Enf.loc[mask, 'Valores de medida'] = 0

    # Heuristic 2: set zero to all the ghost rows created before the first original row. This is made 
    # to eliminate the wrongly created rows which came from the entry of "Hospitais de Campanha" 
    array_indices = np.where(df_hosp_Enf['Original'] == 1)
    first_occur = array_indices[0][0]  #get the index of the first original (available) row
    if (first_occur != 0):
        idxs_first_period = pd.date_range(d_first, periods = first_occur, freq = "D") #list of date idxs
        #print(first_occur)
        #print(idxs_first_period)
        df_hosp_Enf.loc[idxs_first_period, 'Valores de medida'] = 0
        
    #print(df_hosp_Enf[['Nome Fantasia','Valores de medida', 'Original']].head)
    #time.sleep(2)

    # Compute the moving average (win=7)
    df_hosp_Enf['mv'] = df_hosp_Enf['Valores de medida'].rolling(7, min_periods = 1).mean()
    #----------------------------------

    # Case 1: 'Pacientes UTI'
    #----------------------------------
    df_hosp_UTI = df.loc[(df['key'] == i) & (df['Nomes de medida'] == hosp_ind[1])]    

    # Set the 'Data' field as index, and reindex in order to add the missing rows from 'idxs_period'
    df_hosp_UTI.set_index('Data', inplace=True)
    df_hosp_UTI.index = pd.DatetimeIndex(df_hosp_UTI.index)
    df_hosp_UTI = df_hosp_UTI.reindex(idxs_period) #expand the rows to get 'idxs_periods'

    # Fill NaN from 'Original' field with 0.0 (i.e., set the row as an artificially created one)
    df_hosp_UTI['Original'] = df_hosp_UTI['Original'].fillna(0)

    # Fill the NaN values of the rows after the original rows, and then, fill the remaining rows 
    # before the first filled row
    df_hosp_UTI = df_hosp_UTI.ffill().bfill()

    # Heuristic 1: set zero to the artificially created rows (Original=0) whose "Valores de medida = 1"
    # (because the current row is a ghost one generated from an original sample with value = 1) 
    mask = (df_hosp_UTI['Original'] == 0) & (df_hosp_UTI['Valores de medida'] == 1)
    df_hosp_UTI.loc[mask, 'Valores de medida'] = 0

    # Heuristic 2: set zero to all the ghost rows created before the first original row. This is made 
    # to eliminate the wrongly created rows which came from the entry of "Hospitais de Campanha" 
    array_indices = np.where(df_hosp_UTI['Original'] == 1)
    first_occur = array_indices[0][0]  #get the index of the first original (available) row
    if (first_occur != 0):
        idxs_first_period = pd.date_range(d_first, periods = first_occur, freq = "D") #list of date idxs
        #print(first_occur)
        #print(idxs_first_period)
        df_hosp_UTI.loc[idxs_first_period, 'Valores de medida'] = 0

    # Compute the moving average (win=7)
    df_hosp_UTI['mv'] = df_hosp_UTI['Valores de medida'].rolling(7, min_periods = 1).mean()
    #----------------------------------

    # Salve both dataframes into a 'Big' Dataframe
    df_final.append(df_hosp_Enf)
    df_final.append(df_hosp_UTI)
    #print(df_hosp_Enf[['Nomes de medida','Valores de medida']])

# Contatenate all the sub-dataframes ('Enfermaria' e 'UTI')
df_final = pd.concat(df_final)

# Drop all rows where 'Nome Fantasia' is empty
df_final = df_final[~df_final['Nome Fantasia'].isnull()]

# Final adjustment (remove 'key' field and convert date-time)
#df_final.set_index('Data', drop=True, inplace=True)
df_final = df_final.drop('key', 1)
df_final.reset_index(inplace=True)
df_final.rename({'index': 'Data'}, axis = 1, inplace = True)

#df.index = [i for i in range(df_final.shape)]

print(df_final.head())

df_final.to_excel("output.xlsx")
