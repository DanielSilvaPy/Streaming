# Projeto de Classificação de Cancelamento de Serviço de Streaming

Este projeto tem como objetivo desenvolver um modelo de classificação capaz de prever se um cliente irá cancelar o serviço de streaming ou não, levando em consideração o seu perfil. O modelo foi testado utilizando vários algoritmos de machine learning, como Regressão Logística e Random Forest, com o intuito de encontrar o modelo que melhor se adapta ao problema de negócio.

## 🎯 Objetivos

- Desenvolver um modelo de classificação para prever o cancelamento de um serviço de streaming com base no perfil do usuário.
- Comparar a performance de diferentes algoritmos de machine learning.
- Usar gráficos e visualizações para auxiliar e enriquecer a análise.
- Documentar as etapas realizadas e justificar as escolhas feitas no desenvolvimento do modelo.

---

## 💻 Etapas de Desenvolvimento

### **Etapa 01) Análise Exploratória dos Dados (Data Understanding)**

Na primeira etapa, a análise exploratória dos dados foi realizada para entender a estrutura do dataset e a distribuição das variáveis.

#### 1. Carregamento dos Dados
- Os dados foram carregados e armazenados em um DataFrame.

#### 2. Descrição Estatística
- Foram utilizadas funções como `.describe()` para entender as estatísticas das variáveis numéricas.

#### 3. Tipos de Dados
- Verificamos os tipos de dados com a função `.info()`.

#### 4. Valores Faltantes
- A quantidade de valores faltantes foi verificada utilizando `.isna().sum()`.

### **Etapa 02) Tratamento dos Dados (Data Preparation)**

Nesta etapa, os dados foram tratados para garantir que o modelo receba as informações no formato adequado.

#### 1. Substituição de Valores NaN
- Substituímos os valores NaN por 0 nas colunas: `Time_on_platform`, `Num_streaming_services`, `Churned`, `Avg_rating`, `Devices_connected`.

#### 2. Remoção de Linhas Nulas
- Linhas com valores nulos nas colunas `Gender`, `Subscription_type`, e `Age` foram removidas.

#### 3. Transformação de Valores de Churn
- Os valores `0` e `1` na coluna `Churned` foram transformados para `No` e `Yes`.

#### 4. Conversão de Valores Float para Inteiro
- As variáveis numéricas que estavam como `float` foram convertidas para `int` para otimizar o modelo.

### **Etapa 03) Modelagem dos Dados - Regressão Logística**

#### 1. Definição das Variáveis X e y
- A variável `X` contém as colunas de características (input) e `y` contém a variável alvo `Churned`.

#### 2. Treinamento do Modelo
- Utilizamos o modelo de **Regressão Logística** e realizamos o treinamento usando a função `.fit()`.

#### 3. Separação em Conjuntos de Treinamento e Teste
- Os dados foram divididos em treinamento e teste com a função `train_test_split()`.

#### 4. Modelagem e Avaliação
- Após treinar o modelo, fizemos previsões no conjunto de teste utilizando `.predict()` e avaliamos o desempenho com métricas como **Acurácia**, **Precisão**, **Recall** e **F1-Score**.

#### 5. Matriz de Confusão
- A matriz de confusão foi plotada usando a função `ConfusionMatrixDisplay`.

### **Etapa 04) Modelagem dos Dados - Tuning**

#### 1. Pré-processamento das Variáveis Categóricas
- As variáveis categóricas foram pré-processadas usando `LabelEncoder` para transformar valores textuais em numéricos.

#### 2. Tuning de Hiperparâmetros com GridSearchCV
- Realizamos o tuning do modelo utilizando `GridSearchCV` para encontrar a melhor combinação de parâmetros de modelos como Regressão Logística.

#### 3. Avaliação do Modelo
- O modelo ajustado foi avaliado utilizando novamente a matriz de confusão e as métricas de desempenho.

### **Etapa 05) Modelagem dos Dados - Random Forest**

#### 1. Montagem do Grid Search
- Realizamos a montagem do grid search para encontrar os melhores parâmetros do modelo de **Random Forest**.

#### 2. Ajuste do Modelo
- O modelo foi treinado com o melhor conjunto de parâmetros encontrados.

#### 3. Avaliação do Modelo
- As métricas de avaliação foram geradas para o modelo de Random Forest, e a matriz de confusão foi novamente plotada para visualização.

#### 4. Comparação de Modelos
- A performance do modelo de Random Forest foi comparada com a Regressão Logística para identificar qual modelo obteve melhor desempenho.

---

## 📊 Resultados e Métricas

### Métricas do Modelo
- **Acurácia**: Taxa de acerto do modelo.
- **Precisão**: Proporção de verdadeiros positivos entre todas as previsões positivas feitas.
- **Recall**: Proporção de verdadeiros positivos entre todos os casos positivos reais.
- **F1-Score**: Média harmônica entre precisão e recall.

### Melhor Modelo
- O modelo de **Random Forest** obteve melhor performance comparado à Regressão Logística, com uma **acurácia de 0.52** no conjunto de teste.

### Matriz de Confusão
- A matriz de confusão foi plotada para cada modelo, permitindo uma visualização das classificações corretas e incorretas feitas pelos modelos.

---

## 🔧 Tecnologias e Bibliotecas

- **Python 3.x**
- **pandas**: Para manipulação de dados.
- **numpy**: Para operações numéricas.
- **matplotlib**: Para visualização de gráficos.
- **scikit-learn**: Para a construção e avaliação de modelos de machine learning.
- **seaborn**: Para visualizações adicionais e mais avançadas.
- **GridSearchCV**: Para busca de hiperparâmetros.

---

## 🚀 Como Rodar o Projeto

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/seu-usuario/projeto-streaming-cancelamento.git
   cd projeto-streaming-cancelamento
