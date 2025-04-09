import pandas as pd

from src.load_data.load_data import loadDatasetProcessed

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def train_logistic_regression():
    dataset = loadDatasetProcessed()  # Carregar o dataset processado
    
    if dataset is not None:
        le = LabelEncoder()

        # Codificando as colunas categóricas
        dataset['Gender'] = le.fit_transform(dataset['Gender'])
        dataset['Subscription_type'] = le.fit_transform(dataset['Subscription_type'])
        dataset['Churned'] = le.fit_transform(dataset['Churned'])

        # Identificar as colunas numéricas para normalizar
        numeric_columns = dataset.select_dtypes(include=['int64']).columns
    
        # Excluir a coluna 'User_id' (não numérica) antes da normalização
        dataset = dataset.drop(columns=['User_id'])

        scaler = MinMaxScaler()

        # Aplicar o MinMaxScaler apenas nas colunas numéricas
        dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

        # Selecionar as colunas de características
        X = dataset[['Age', 'Gender', 'Time_on_platform', 'Devices_connected', 'Subscription_type', 'Num_streaming_services', 'Num_active_profiles', 'Avg_rating']]
        Y = dataset['Churned']  # Alvo (Churned)

        # Dividir os dados em treino e teste
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

        # Treinar o modelo de regressão logística
        model = LogisticRegression(class_weight='balanced').fit(x_train, y_train)

        # Fazer previsões no conjunto de teste
        predict = model.predict(x_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, predict)
        precision = precision_score(y_test, predict)
        recall = recall_score(y_test, predict)
        f1 = f1_score(y_test, predict)

        # Exibir as métricas
        print(f'Acurácia: {accuracy}')
        print(f'Precisão: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')

        # Gerar a matriz de confusão
        cm = confusion_matrix(y_test, predict)
        print(f'Matriz de Confusão:\n{cm}')

        # Exibir a matriz de confusão usando ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classe 0', 'Classe 1'])
        disp.plot(cmap='Blues')  # Usando um mapa de cores azul
        plt.show()

    return None

def tunning():
    # Carregar o dataset processado
    dataset = loadDatasetProcessed()
    
    if dataset is not None:
        le = LabelEncoder()

        # Codificando as colunas categóricas
        dataset['Gender'] = le.fit_transform(dataset['Gender'])
        dataset['Subscription_type'] = le.fit_transform(dataset['Subscription_type'])
        dataset['Churned'] = le.fit_transform(dataset['Churned'])

        # Identificar as colunas numéricas para normalizar
        numeric_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
        
        # Excluir a coluna 'User_id' (não numérica) antes da normalização
        dataset = dataset.drop(columns=['User_id'])

        scaler = MinMaxScaler()
        # Aplicar o MinMaxScaler nas colunas numéricas
        dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

        # Selecionar as colunas de características
        X = dataset[['Age', 'Gender', 'Time_on_platform', 'Devices_connected', 'Subscription_type', 'Num_streaming_services', 'Num_active_profiles', 'Avg_rating']]
        Y = dataset['Churned']  # Alvo (Churned)

        # Dividir os dados em treino e teste
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

        # Definir o modelo de regressão logística
        model = LogisticRegression(class_weight='balanced')

        # Definir a grade de parâmetros a ser testada
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularização
            'solver': ['liblinear', 'lbfgs'],  # Algoritmos de otimização
            'penalty': ['l2'],  # Tipo de penalização
            'max_iter': [100, 200, 300]  # Número de iterações
        }

        # Inicializar o GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

        # Ajustar o modelo com o GridSearchCV
        grid_search.fit(x_train, y_train)

        # Melhor combinação de parâmetros
        print(f'Melhores parâmetros: {grid_search.best_params_}')
        print(f'Acurácia no conjunto de teste: {grid_search.score(x_test, y_test)}')

        # Avaliar o modelo com os melhores parâmetros
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        print(classification_report(y_test, y_pred))

        # Gerar a matriz de confusão
        cm = confusion_matrix(y_test, y_pred)

        # Exibir a matriz de confusão
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classe 0', 'Classe 1'])
        disp.plot(cmap='Blues')  # cmap define as cores do gráfico
        plt.show()

    return None

def train_random_forest():
    dataset = loadDatasetProcessed()  # Carregar o dataset processado
    
    if dataset is not None:
        le = LabelEncoder()

        # Codificando as colunas categóricas
        dataset['Gender'] = le.fit_transform(dataset['Gender'])
        dataset['Subscription_type'] = le.fit_transform(dataset['Subscription_type'])
        dataset['Churned'] = le.fit_transform(dataset['Churned'])

        # Identificar as colunas numéricas para normalizar
        numeric_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
    
        # Excluir a coluna 'User_id' (não numérica) antes da normalização
        dataset = dataset.drop(columns=['User_id'])

        # Normalizar as colunas numéricas
        scaler = MinMaxScaler()
        dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

        # Selecionar as colunas de características
        X = dataset[['Age', 'Gender', 'Time_on_platform', 'Devices_connected', 'Subscription_type', 'Num_streaming_services', 'Num_active_profiles', 'Avg_rating']]
        Y = dataset['Churned']  # Alvo (Churned)

        # Dividir os dados em treino e teste
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

        # Definir o modelo de Random Forest
        model = RandomForestClassifier(class_weight='balanced', random_state=42)

        # Definir a grade de parâmetros a ser testada
        param_grid = {
            'n_estimators': [50, 100, 200],  # Número de árvores
            'max_depth': [10, 20, None],  # Profundidade máxima
            'min_samples_split': [2, 5, 10],  # Número mínimo de amostras para dividir um nó
            'min_samples_leaf': [1, 2, 4],  # Número mínimo de amostras por folha
            'bootstrap': [True, False]  # Usar bootstrap ou não
        }

        # Inicializar o GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

        # Ajustar o modelo com o GridSearchCV
        grid_search.fit(X_train, Y_train)

        # Melhor combinação de parâmetros
        print(f'Melhores parâmetros: {grid_search.best_params_}')
        
        # Melhor modelo ajustado
        best_model = grid_search.best_estimator_

        # Fazer previsões no conjunto de teste
        y_pred = best_model.predict(X_test)
        
        # Calcular as métricas
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)

        # Exibir as métricas
        print(f'Acurácia: {accuracy}')
        print(f'Precisão: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')

        # Gerar a matriz de confusão
        cm = confusion_matrix(Y_test, y_pred)
        print(f'Matriz de Confusão:\n{cm}')

        # Exibir a matriz de confusão usando ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classe 0', 'Classe 1'])
        disp.plot(cmap='Blues')  # Usando um mapa de cores azul
        plt.show()

    return None