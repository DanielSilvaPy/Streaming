import sys
import os
import pandas as pd

# Adiciona o diretório raiz ao sys.path para que o Python consiga encontrar a pasta 'plots'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importando as funções corretamente
from load_data.load_data import loadDatasetRaw, loadDatasetProcessed
from data_preprocessing.data_preprocessing import fill_missing_values, drop_columns_with_nulls, transform_churned_values, convert_floats_to_int
from utils.save_data import save_data
from models.model import train_logistic_regression, tunning, train_random_forest

def main():
   # Ajuste para exibir todas as colunas
   pd.set_option('display.max_columns', None)
   dataset = loadDatasetRaw()

   # Etapa (01) Análise exploratória dos dados (Data Understanding)
   print(dataset.info())
   print(dataset.describe())
   print(dataset.isna().sum())

   # Etapa (02) Tratamento dos Dados (Data Preparation)
   dataset = fill_missing_values(dataset)
   dataset = drop_columns_with_nulls(dataset)
   dataset = transform_churned_values(dataset)
   dataset = convert_floats_to_int(dataset)

   save_data(dataset, 'excel', 'data/processed')

   train_logistic_regression()
   tunning()

# Executando o código principal
if __name__ == "__main__":
    main()
