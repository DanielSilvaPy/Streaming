import pandas as pd

def loadDatasetRaw():
    try:
        dataset = pd.read_csv('C:/Users/danie/OneDrive/Documentos/PyVisualCode/Streaming/data/raw/streaming_data.csv')
        return dataset
    except FileNotFoundError:
        print('Não foi possível carregar o dataset.')
        return None
    
def loadDatasetProcessed():
    try:
        dataset = pd.read_excel('C:/Users/danie/OneDrive/Documentos/PyVisualCode/Streaming/data/processed/datasetclean.xlsx')
        return dataset
    except FileNotFoundError:
        print('Não foi possívl carregar o dataset.')
        return None