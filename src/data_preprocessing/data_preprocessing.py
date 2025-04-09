import pandas as pd
from load_data.load_data import loadDatasetRaw
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# Função para apagar as linhas nulas do dataset
def remove_nulls(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Apgar as linhas nulas do dataset.'''
    if dataset is not None:
        return dataset.dropna()
    return None

# Função para apagar as linhas duplicadas.
def drop_duplicates(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Remove as linhas duplicadas do DataFrame.'''
    if dataset is not None:
        dataset.duplicated()
    return None


# Função para substituir NaN por 0 nas colunas especificadas
def fill_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Substituir NaN por 0 nas colunas especificadas.'''
    if dataset is not None:
        dataset['Time_on_platform'] = dataset['Time_on_platform'].fillna(0)
        dataset['Avg_rating'] = dataset['Avg_rating'].fillna(0)
        dataset['Devices_connected'] = dataset['Devices_connected'].fillna(0)
        dataset['Churned'] = dataset['Churned'].fillna(0)
        dataset['Num_streaming_services'] = dataset['Num_streaming_services'].fillna(0)
    return dataset

# Função para dropar linhas nulas nas colunas especificadas
def drop_columns_with_nulls(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Dropar linhas nulas nas colunas 'Gender', 'Subscription_type' e 'Age'.'''
    if dataset is not None:
        dataset = dataset.dropna(subset=['Gender', 'Subscription_type', 'Age'])
    return dataset

# Função para transformar valores 'Churned' de 0 e 1 para 'No' e 'Yes'
def transform_churned_values(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Transformar valores 0 e 1 de 'Churned' para 'No' e 'Yes'.'''
    if dataset is not None:
        dataset['Churned'] = dataset['Churned'].replace({0: 'No', 1: 'Yes'})
        return dataset
    return None

def transformType(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset is not None:
        dataset = dataset.astype(columns={'Age': 'int', 'Time_on_platform': 'int', 'Devices_connected': 'int', 'Num_streaming_services': 'int', 'Avg_rating': 'int'})
        return dataset
    return None

# Função para converter valores float em inteiros para as colunas especificadas
def convert_floats_to_int(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Transformar colunas de valores float em inteiros.'''
    if dataset is not None:
        # Usando astype para converter várias colunas de uma vez
        dataset = dataset.astype({
            'Age': 'int', 
            'Time_on_platform': 'int', 
            'Devices_connected': 'int', 
            'Num_streaming_services': 'int', 
            'Num_active_profiles': 'int', 
            'Avg_rating': 'int'
        }, errors='ignore')  # 'ignore' para evitar erro em caso de valores incompatíveis
        return dataset
    return None
