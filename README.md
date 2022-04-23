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
 
As colunas aproveitadas são descritas abaixo:
 
* `[Fe/H]`: Metalicidade 
* `logg`: Log da gravidade da superfície 
* `Teff`: Temperatura efetiva 
 
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

## Chamando os dados
 
```bash
"""
Chamando o dataset só com gigantes vermelhas
"""
Gigantes_vermelhas = pd.read_csv("2_parte_RGB_HeB.txt", sep = "|", header = None)
Gigantes_vermelhas.columns = ["KIC", "Teff", "e_Teff", "logg", "e_logg", "[Fe/H]", 
                  "e_[Fe/H]", "NoCorM", "e_NoCorM", "NoCorR", "e_NoCorR",
                  "RGBcorM", "e_RGBcorM", "RGBcorR", "e_RGBcorR", "ClcorM", 
                  "e_ClcorM", "ClcorR", "e_ClcorR", "Phase"]
"""
Chamando o dataset só com estrelas frias do tipo Sol
"""
Estrelas_do_tipo_Sol = pd.read_csv("age_prediction.txt", sep = "|", header = None)
Estrelas_do_tipo_Sol.columns = ["Star", "Teff", "e_Teff", "logg",
               "e_logg", "Vt", "e_Vt", "[Fe/H]", "e_[Fe/H]",
               "Mass", "e_Mass", "Age", "e_Age"]
``` 
 
## Pré-processamento de dados
 
```bash
"""
A nossa classificação vai ser com base em três variáveis preditoras principais: Teff / log(g) / [Fe/H] 
""" 
Gigantes_vermelhas.drop(["KIC", "NoCorM", "e_NoCorM", "NoCorR", "e_NoCorR",
                  "RGBcorM", "e_RGBcorM", "RGBcorR", "e_RGBcorR", "ClcorM", 
                  "e_ClcorM", "ClcorR", "e_ClcorR", "Phase", "e_Teff", "e_logg", "e_[Fe/H]"], axis = 1, inplace = True)
Estrelas_do_tipo_Sol.drop(["Star", "Vt", "e_Vt",
                          "Mass", "e_Mass", "Age", "e_Age", "e_Teff", "e_logg", "e_[Fe/H]"], axis = 1, inplace = True)
"""
Agora, vamos embaralhar as linhas dos dois Dataframes... 
""" 
Gigantes_vermelhas_ = Gigantes_vermelhas.sample(frac=1).reset_index(drop = True)
Estrelas_do_tipo_Sol_ = Estrelas_do_tipo_Sol.sample(frac=1).reset_index(drop = True)
Gigantes_vermelhas = pd.DataFrame(Gigantes_vermelhas_, columns = Gigantes_vermelhas.columns)
Estrelas_do_tipo_Sol = pd.DataFrame(Estrelas_do_tipo_Sol_, columns = Estrelas_do_tipo_Sol.columns)
"""
Note que há uma disparidade muito grande entre o número de linhas dos dois Dataframes; 
""" 
print(f"Shape_Gigantes_vermelhas = {Gigantes_vermelhas.shape}")
print(f"Shape_Estrelas_do_tipo_Sol = {Estrelas_do_tipo_Sol.shape}")
"""
Shape_Gigantes_vermelhas = (16094, 3)
Shape_Estrelas_do_tipo_Sol = (451, 3) 
Vamos pegar apenas 451 linhas do Dataframe das gigantes vermelhas 
""" 
Gigantes_vermelhas = Gigantes_vermelhas.loc[0:450]
print(f"Shape_Gigantes_vermelhas = {Gigantes_vermelhas.shape}")
print(f"Shape_Estrelas_do_tipo_Sol = {Estrelas_do_tipo_Sol.shape}") 
"""
Shape_Gigantes_vermelhas = (451, 3)
Shape_Estrelas_do_tipo_Sol = (451, 3) 
""" 
``` 
 
Pronto, agora vamos adicionar a variável target aos Dataframes.
 
Gigante vermelha = 0
 
Estrela do tipo Sol = 1

```bash
target = []
for i in range(0, 451):
    target.append(0)
target = pd.DataFrame(target, columns = ["target"])
Gigantes_vermelhas = pd.concat([Gigantes_vermelhas, target], axis = 1)
target = []
for i in range(0, 451):
    target.append(1)
target = pd.DataFrame(target, columns = ["target"])
Estrelas_do_tipo_Sol = pd.concat([Estrelas_do_tipo_Sol, target], axis = 1)
"""
Hora de concatenar os dois Dataframes... 
""" 
Concatenado_GV_SSOL = pd.concat([Gigantes_vermelhas, Estrelas_do_tipo_Sol], axis = 0)
Concatenado_GV_SSOL = Concatenado_GV_SSOL.sample(frac = 1).reset_index(drop = True)
Concatenado_GV_SSOL = pd.DataFrame(Concatenado_GV_SSOL, columns = Estrelas_do_tipo_Sol.columns)  
``` 

## Análise dos dados
 
Usando três variáveis preditoras para fazer a classificação: Teff / Log(g) / [Fe/H] formaram-se dois clusters de dados visivelmente separáveis. Isso já é um indício que os algoritmos de ML irão se dar muito bem nessa classificação. 
 
<p align="center"> 

<img src = "https://user-images.githubusercontent.com/93550626/164868931-e76ce1b6-6fbb-46e8-87f4-9ab65e7e6807.jpg" width = 400 height = 400> 
<img src = "https://user-images.githubusercontent.com/93550626/164868937-a77473f5-d117-4620-bc1d-0d552111a1d6.jpg" width = 400 height = 400>

<img src = "https://user-images.githubusercontent.com/93550626/164873300-df5c8aa9-fdb0-4b1a-ae5f-bc48c311ab3b.jpg" width = 900 height = 400> 

<p> 
 
 
 
 
 


