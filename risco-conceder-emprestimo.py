#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px


# In[29]:


base_creditos = pd.read_csv('base-dados/credit_data.csv')
base_creditos
# clientid  |income      |  age     | loan                       |  default
# cliente   |renda anual |  idade   | divida que a pessoa possui |  pagou(0) ou não(1) o empréstimo


# ## Tratando os dados da tabela

# In[3]:


# mostrando que a tabela tinha idades negativas
base_creditos.loc[base_creditos['age'] < 0]


# In[4]:


# nesse caso eu vou por a média das idades da tabela nos valores negativos
media_idades = base_creditos['age'].loc[base_creditos['age'] > 0].mean()
base_creditos.loc[base_creditos['age'] < 0, 'age'] = media_idades 


# In[5]:


# mostrando que a tabela tem valores faltantes
base_creditos.loc[pd.isnull(base_creditos['age'])]


# In[6]:


# colocando a média das idades nas linhas sem idades
base_creditos['age'].fillna(base_creditos['age'].mean(), inplace=True)


# ## Dividindo a tebela entre os atributos previsores  e as classes (objetivos)

# In[7]:


X_credito = base_creditos.iloc[:, 1:4].values  # transformando as colunas 1 a 3 em uma lista numpy (array)


# In[8]:


y_credito = base_creditos.iloc[:, 4].values   # transformando a coluna 4 em uma lista numpy(array) (objetivo)


# #### colocando todos os valores das colunas de previsores numa mesma escala

# In[9]:


scaler_credit = StandardScaler()
X_credito = scaler_credit.fit_transform(X_credito)


# ## Dividindo a base com 75% para treino no algoritimo de machine learning e 25% para testes

# In[10]:


X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X_credito, y_credito, test_size=0.25, random_state=0)


# ## Iniciando a criação e treinamento (random forest)

# In[18]:


random_forest_credit = RandomForestClassifier(criterion='entropy', n_estimators=40, random_state=0)
random_forest_credit.fit(X_treinamento, y_treinamento)


# In[36]:


previsoes = random_forest_credit.predict(X_teste)
#previsoes, y_teste


# In[22]:


accuracy_score(y_teste, previsoes)  # totalizando 98,40% de acerto


# In[24]:


print(classification_report(y_teste, previsoes))


# In[32]:


grafico = px.scatter_matrix(base_creditos, dimensions=['age', 'income'], color='default')
grafico.show()
