# Diferenciando estrelas do tipo Sol de gigantes vermelhas usando ML

<h1>

```python
print("Introdução")
```  
 
</h1>  

Um dos motivos principais para a existência de vida na Terra é <b>a gama de condições favoráveis que o Sol propicia ao nosso planeta</b>. Por exemplo, em certa medida, a luz do Sol é um parâmetro básico de qualquer vida na botânica, pois as substâncias orgânicas que nutrem uma planta são produzidas por meio de <b>fotossíntese</b> e, nesse processo, a principal fonte de energia é a luz do Sol.

<p align="center">
 
<img src = "https://user-images.githubusercontent.com/93550626/164562407-7fafce67-f4c7-44c9-88e4-d336906ebfda.jpg" width = 400 height = 400>
<img src = "https://user-images.githubusercontent.com/93550626/164562960-26b727de-c2ff-43c3-b8a1-20ec224c5701.JPG" width = 400 height = 400>
 
<p>

Sendo assim, encontrar estrelas do tipo Sol é de extrema importância para a procura de vida fora da Terra. Mas, afinal o que são estrelas do tipo Sol? De uma forma simples, são estrelas que <b>estão na sequência principal com um índice de cor B-V entre 0.48 e 0.80</b>. O sol possui um índice B-V igual a 0.65. 

<p align="center">
 
<img src = "https://user-images.githubusercontent.com/93550626/164580474-38f935b3-1f69-4a09-9692-8f4da14e9123.jpg" width = 400 height = 400> 
<img src = "https://user-images.githubusercontent.com/93550626/164580407-365bf683-a25b-40d7-8236-ebb9b175f215.jpg" width = 400 height = 400> 
 
<p>

 
##

<h1>

```python
print("Objetivos")
```  
 
</h1>

* Esse é um projeto inicial que pretende <b>destinguir estrelas frias do tipo solar de gigantes vermelhas</b> usando Machine Learning.


##

<h1>

```python
print("Procedimentos")
```
 
## Conjunto de dados original
 
Foram usados dois conjuntos de dados diferentes nesse trabalho, um para cada tipo de estrela. A fonte dos dados foi o repositório de catálogos <>VizieR</b>.

Para as gigantes vermelhas, no VizieR, nós temos a 2º tabela desse repositório: <a href="https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/236/42#/browse">Asteroseismology of ~16000 Kepler red giants : J/ApJS/236/42</a>
 
Para as estrelas do tipo Sol, nós temos a 3º tabela desse repositório: <a href="https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/555/A150#/browse">Physical parameters of cool solar-type stars : J/A+A/555/A150</a>
 
## Importação das bibliotecas
 
```bash
import numpy as np # Importando o numpy para trabalhar com matrizes e etc.
import pandas as pd # Importação do pandas para trabalhar com dados.
from pandas_profiling import ProfileReport # Importando o ProfileReport para fazer uma análise geral do dataset
"""
Importações para trabalhar com gráficos
"""
import seaborn as sbn
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator 
from matplotlib.font_manager import FontProperties
"""
Importações para Machine Learning
"""
"""
train_test_split: Realiza um split nos dados entre treino e teste.
GridSearchCV: Trabalha com ajuste de hiperparâmetros para os algoritmos de ML.
"""
from sklearn.model_selection import train_test_split, GridSearchCV
"""
Ignorar alguns warnings que não afetam o código
"""
import warnings
warnings.filterwarnings("ignore")
"""
Importação das métricas de avaliação de um modelo;
"""
from sklearn.metrics import (classification_report, # Report geral
                             accuracy_score, # Acurácia
                             roc_auc_score,
                             roc_curve, # Curva roc
                             confusion_matrix) # Matriz de confusão
"""
MinMaxScaler: Realiza a normalização dos dados
"""
from sklearn.preprocessing import MinMaxScaler
"""
Regressão logística
"""
from sklearn.linear_model import LogisticRegression
"""
Algoritmo dos K vizinhos mais próximos
"""
from sklearn.neighbors import KNeighborsClassifier
"""
Importando do Naive Bayes o algoritmo GaussianNB
"""
from sklearn.naive_bayes import GaussianNB
"""
Algoritmos de aprendizado não supervisionado: KMeans
"""
from sklearn.cluster import KMeans
"""
Importação dos algoritmos ensemble
"""
from sklearn.ensemble import (RandomForestClassifier, 
                             ExtraTreesClassifier, 
                             AdaBoostClassifier, 
                             GradientBoostingClassifier)
"""
SVC: Support vector machine
"""
from sklearn.svm import SVC
"""
Parte das importações para a rede neural
"""
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout 
```
 
 
 
 
 
 


