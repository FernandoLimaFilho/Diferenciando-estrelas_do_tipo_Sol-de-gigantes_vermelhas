#!/usr/bin/env python
# coding: utf-8

# # $\color{orange}{\textbf{Introdução}}$

# $\color{gray}{\textbf{Um dos motivos principais para a existência de vida na Terra é a gama de condições favoráveis}}$
# $\color{gray}{\textbf{que o Sol propicia ao nosso planketa. Sendo assim, nota-se que encontrar estrelas do tipo Sol é de}}$
# $\color{gray}{\textbf{extrema importância para a procura de vida fora da Terra.}}$

# # $\color{orange}{\textbf{Objetivos}}$

# $\color{gray}{\textbf{Esse é um projeto inicial que pretende destinguir estrelas frias do tipo solar de gigantes vermelhas}}$
# $\color{gray}{\textbf{usando Machine Learning.}}$

# # $\color{orange}{\textbf{Importação das bibliotecas}}$

# In[86]:


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


# # $\color{orange}{\textbf{Chamando os dados}}$

# In[87]:


"""
Chamando o dataset só com gigantes vermelhas
"""
Gigantes_vermelhas = pd.read_csv("2_parte_RGB_HeB.txt", sep = "|", header = None)
Gigantes_vermelhas.columns = ["KIC", "Teff", "e_Teff", "logg", "e_logg", "[Fe/H]", 
                  "e_[Fe/H]", "NoCorM", "e_NoCorM", "NoCorR", "e_NoCorR",
                  "RGBcorM", "e_RGBcorM", "RGBcorR", "e_RGBcorR", "ClcorM", 
                  "e_ClcorM", "ClcorR", "e_ClcorR", "Phase"]
"""
Mostra na tela a parte superior do DF "Gigantes_vermelhas"
"""
pd.set_option("display.max_columns", len(Gigantes_vermelhas.columns))
Gigantes_vermelhas.head(14)


# In[88]:


"""
Chamando o dataset só com estrelas frias do tipo Sol
"""
Estrelas_do_tipo_Sol = pd.read_csv("age_prediction.txt", sep = "|", header = None)
Estrelas_do_tipo_Sol.columns = ["Star", "Teff", "e_Teff", "logg",
               "e_logg", "Vt", "e_Vt", "[Fe/H]", "e_[Fe/H]",
               "Mass", "e_Mass", "Age", "e_Age"]
"""
Mostra na tela a parte superior do DF "Estrelas_do_tipo_Sol"
"""
pd.set_option("display.max_columns", len(Estrelas_do_tipo_Sol.columns))
Estrelas_do_tipo_Sol.head(14)


# # $\color{orange}{\textbf{Pré-processamento de dados}}$

# $\color{gray}{\textbf{A nossa classificação vai ser com base em três variáveis preditoras principais: Teff / log(g) / [Fe/H]}}$

# In[89]:


Gigantes_vermelhas.drop(["KIC", "NoCorM", "e_NoCorM", "NoCorR", "e_NoCorR",
                  "RGBcorM", "e_RGBcorM", "RGBcorR", "e_RGBcorR", "ClcorM", 
                  "e_ClcorM", "ClcorR", "e_ClcorR", "Phase", "e_Teff", "e_logg", "e_[Fe/H]"], axis = 1, inplace = True)
Estrelas_do_tipo_Sol.drop(["Star", "Vt", "e_Vt",
                          "Mass", "e_Mass", "Age", "e_Age", "e_Teff", "e_logg", "e_[Fe/H]"], axis = 1, inplace = True)


# $\color{gray}{\textbf{Agora, vamos embaralhar as linhas dos dois Dataframes...}}$

# In[90]:


Gigantes_vermelhas_ = Gigantes_vermelhas.sample(frac=1).reset_index(drop = True)
Estrelas_do_tipo_Sol_ = Estrelas_do_tipo_Sol.sample(frac=1).reset_index(drop = True)
Gigantes_vermelhas = pd.DataFrame(Gigantes_vermelhas_, columns = Gigantes_vermelhas.columns)
Estrelas_do_tipo_Sol = pd.DataFrame(Estrelas_do_tipo_Sol_, columns = Estrelas_do_tipo_Sol.columns)


# $\color{gray}{\textbf{Note que há uma disparidade muito grande entre o número de linhas dos dois Dataframes;}}$

# In[91]:


print(f"Shape_Gigantes_vermelhas = {Gigantes_vermelhas.shape}")
print(f"Shape_Estrelas_do_tipo_Sol = {Estrelas_do_tipo_Sol.shape}")


# $\color{gray}{\textbf{Vamos pegar apenas 451 linhas do Dataframe das gigantes vermelhas}}$

# In[92]:


Gigantes_vermelhas = Gigantes_vermelhas.loc[0:450]


# In[93]:


print(f"Shape_Gigantes_vermelhas = {Gigantes_vermelhas.shape}")
print(f"Shape_Estrelas_do_tipo_Sol = {Estrelas_do_tipo_Sol.shape}")


# $\color{gray}{\textbf{Pronto, agora vamos adicionar a variável target aos Dataframes.}}$
# 
# $\color{gray}{\textbf{Gigante vermelha = 0}}$
# 
# $\color{gray}{\textbf{Estrela do tipo Sol = 1}}$

# In[94]:


target = []
for i in range(0, 451):
    target.append(0)
target = pd.DataFrame(target, columns = ["target"])
Gigantes_vermelhas = pd.concat([Gigantes_vermelhas, target], axis = 1)
Gigantes_vermelhas.head()


# In[95]:


target = []
for i in range(0, 451):
    target.append(1)
target = pd.DataFrame(target, columns = ["target"])
Estrelas_do_tipo_Sol = pd.concat([Estrelas_do_tipo_Sol, target], axis = 1)
Estrelas_do_tipo_Sol.head()


# $\color{gray}{\textbf{Hora de concatenar os dois Dataframes...}}$

# In[96]:


Concatenado_GV_SSOL = pd.concat([Gigantes_vermelhas, Estrelas_do_tipo_Sol], axis = 0)
Concatenado_GV_SSOL = Concatenado_GV_SSOL.sample(frac = 1).reset_index(drop = True)
Concatenado_GV_SSOL = pd.DataFrame(Concatenado_GV_SSOL, columns = Estrelas_do_tipo_Sol.columns)
Concatenado_GV_SSOL.head(30)


# # $\color{orange}{\textbf{Dtypes}}$

# In[97]:


Concatenado_GV_SSOL.dtypes


# # $\color{orange}{\textbf{Análise dos dados}}$

# In[98]:


ProfileReport(Concatenado_GV_SSOL)


# In[99]:


"""
Criação da primeira fonte de texto
"""
Font1 = {"family":"serif", # Family da fonte
         "weight":"normal", # Peso da fonte
         "color": "gray", # cor da fornte
         "size": 12.4} # size da fonte
"""
Plotando histogramas para cada variável característica
"""
for i in Concatenado_GV_SSOL.drop(["target"], axis = 1).columns:
    """
    Alocando a figura
    """
    fig, ax = plt.subplots(figsize = (9, 7))
    """
    Plot do gráfico
    """
    sbn.distplot(Concatenado_GV_SSOL.drop(["target"], axis = 1)[i], color = "orange")
    plt.grid(False)
    """
    Redefinição da grossura dos eixos e da cor dos mesmos
    """
    for axis in ["left", "right", "top", "bottom"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("gray")
    """
    Trabalha com os ticks do gráfico
    """     
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = "both", direction = "in", labelcolor = "gray", labelsize = 12.4)
    ax.tick_params(which = "minor", direction = "in", width = 2, color = "gray")
    ax.tick_params(which = "major", direction = "in", color = "gray", length=3.4, width = 2)
    """
    Labels
    """
    ax.set_ylabel("Densidade", fontdict = Font1)
    ax.set_xlabel(f"{i}", fontdict = Font1)
    """
    Tudo em negrito
    """
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    """
    Fundo branco
    """
    fig.patch.set_facecolor("white")
    Cor_fundo = plt.gca()
    Cor_fundo.set_facecolor("white")
    Cor_fundo.patch.set_alpha(1)
    fig.patch.set_facecolor("white")
    """
    Mostrar gráfico
    """
    plt.show()


# In[100]:


"""
Plot dos histogramas para cada variável
"""
for i in Concatenado_GV_SSOL.drop(["target"], axis = 1).columns:
    """
    Alocando a figura
    """
    fig, ax = plt.subplots(figsize=(11,4))
    """
    Plot do gráfico
    """
    Concatenado_GV_SSOL.drop(["target"], axis = 1).boxplot(column = i, grid = False, fontsize=12)
    fig.patch.set_facecolor("white")
    for axis in ["left", "right", "top", "bottom"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("gray")
    ax.xaxis.set_minor_locator(AutoMinorLocator())    
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = "both", direction = "in", labelcolor = "gray", labelsize = 12, bottom = False)
    ax.tick_params(which = "minor", direction = "in", width = 2, color = "gray", bottom = False)
    ax.tick_params(which = "major", direction = "in", color = "gray", length=3.4, width = 2, bottom = False)
    """
    Tudo em negrito
    """
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    """
    Fundo branco
    """
    fig.patch.set_facecolor("white")
    Cor_fundo = plt.gca()
    Cor_fundo.set_facecolor("white")
    Cor_fundo.patch.set_alpha(1)
    fig.patch.set_facecolor("white")
    plt.show()


# In[101]:


ax, fig = plt.subplots(figsize = (9, 7))
"""
Matriz de correlação entre as variáveis mostrada na forma de um mapa de calor
"""
sbn.heatmap(Concatenado_GV_SSOL.drop(["target"], axis = 1).corr(), # Matriz de correlação
            annot = True, # Anotar p = True
            vmin = -1, # p min
            vmax = 1, # p max
            cmap = "Oranges", # Colormap
            linewidths = 2, # width da linha de controno entre as células do mapa de calor
            linecolor = "white", # cor de tais linhas
            annot_kws = {"size": 13.2}) # size dos números no heatmap
"""
Mudando o size da fonte dos labels
"""
sbn.set(font_scale=1.15)
"""
Tudo em negrito
"""
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
"""

"""
fig.patch.set_facecolor("white")
"""
Mostrar gráfico
"""
plt.show()


# In[102]:


"""
Criação da 2º fonte
"""
Font2 = FontProperties(family = "serif",
                      weight = "bold",
                      style = "normal",
                      size = 12)
fig = plt.figure(figsize = (7, 10)) # Alocar a figura
ax = fig.add_subplot(projection = "3d") # plot 3d
ax.scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["Teff"], 
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["logg"],
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["[Fe/H]"], c = "darkred", label = "Gigantes vermelhas")
ax.scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["Teff"], 
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["logg"],
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["[Fe/H]"], c = "orange", label = "Estrelas frias do tipo Solar")
ax.view_init(20, -50) # Ângulo de visão
"""
Labels
"""
ax.set_xlabel("Teff $[K]$", fontdict = Font1)
ax.set_ylabel("Log(g) $[cm/s^{2}]$", fontdict = Font1)
ax.set_zlabel("[Fe/H] $[Sun]$", fontdict = Font1)
"""
Fundo branco
"""
fig.patch.set_facecolor("white")
Cor_fundo = plt.gca()
Cor_fundo.set_facecolor("white")
Cor_fundo.patch.set_alpha(1)
"""
Legenda
"""
plt.legend(frameon = False, prop = Font2, labelcolor = "gray")
"""
Mostrar o gráfico
"""
plt.show()


# In[103]:


Font2 = FontProperties(family = "serif",
                      weight = "bold",
                      style = "normal",
                      size = 12)
fig = plt.figure(figsize = (7, 10))
ax = fig.add_subplot(projection = "3d")
ax.scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["Teff"], 
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["logg"],
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["[Fe/H]"], c = "darkred", label = "Gigantes vermelhas")
ax.scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["Teff"], 
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["logg"],
           Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["[Fe/H]"], c = "orange", label = "Estrelas frias do tipo Solar")
ax.view_init(9, 10)
ax.set_xlabel("Teff $[K]$", fontdict = Font1)
ax.set_ylabel("Log(g) $[cm/s^{2}]$", fontdict = Font1)
ax.set_zlabel("[Fe/H] $[Sun]$", fontdict = Font1)
fig.patch.set_facecolor("white")
Cor_fundo = plt.gca()
Cor_fundo.set_facecolor("white")
Cor_fundo.patch.set_alpha(1)
plt.legend(frameon = False, prop = Font2, labelcolor = "gray")
plt.show()


# $\color{gray}{\textbf{Na visualização 3d percebe-se a formação de aglomerados bem definidos de dados.}}$

# In[104]:


"""
Criação das 3º e 4º fontes
"""
Font3 = {"family":"serif", "weight":"normal", "color": "gray", "size": 18}
Font4 = FontProperties(family = "serif",
                      weight = "bold",
                      style = "normal",
                      size = 18)
fig, axs = plt.subplots(nrows = 1, # 1 linha
                        ncols = 2, # 2 colunas
                        figsize = (19, 8)) # tamanho da figura
axs[0].scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["logg"], Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["Teff"], s = 30, c = "darkred", label = "Gigantes vermelhas")
axs[0].scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["logg"], Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["Teff"], s = 30, c = "orange", label = "Estrelas frias do tipo Solar")
axs[1].scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["[Fe/H]"], Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 0]["Teff"], s = 30, c = "darkred", label = "Gigantes vermelhas")
axs[1].scatter(Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["[Fe/H]"], Concatenado_GV_SSOL[Concatenado_GV_SSOL["target"] == 1]["Teff"], s = 30, c = "orange", label = "Estrelas frias do tipo Solar")
axs[0].set_xlabel("Log(g) $[cm/s^{2}]$", fontdict = Font3)
axs[1].set_xlabel("[Fe/H] $[Sun]$", fontdict = Font3)
axs[0].set_ylabel("Teff [$K$]", fontdict = Font3)
for i in range(0, 2):
    for axis in ["left", "right", "top", "bottom"]:
        axs[i].spines[axis].set_linewidth(2)
        axs[i].spines[axis].set_color("gray")
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].yaxis.set_minor_locator(AutoMinorLocator())
    axs[i].tick_params(axis = "both", direction = "in", labelcolor = "gray", labelsize = 18, top = True, right = True, left = True, bottom = True)
    axs[i].tick_params(which='minor', direction = "in", length=2, color='gray', width = 2, top = True, right = True, left = True, bottom = True)
    axs[i].tick_params(which='major', direction = "in", color='gray', length=3.4, width = 2, top = True, right = True, left = True, bottom = True)
fig.tight_layout()
axs[0].legend(frameon = False, prop = Font4, labelcolor = "gray")
axs[1].legend(frameon = False, prop = Font4, labelcolor = "gray")
plt.show()


# # $\color{orange}{\textbf{Split dos dados}}$

# In[105]:


"""
x: DF com apenas variáveis preditoras
"""
x = Concatenado_GV_SSOL.drop(["target"], axis = 1)
Norm = MinMaxScaler(feature_range = (0, 1))
"""
Normalizar x
"""
x_norm = Norm.fit_transform(x)
x_norm = pd.DataFrame(x_norm, columns = x.columns)
"""
y: Série com a variável target 
"""
y = Concatenado_GV_SSOL["target"]
"""
y_neural_network: y para a rede neural
"""
y_neural_network = to_categorical(y)
"""
Splits dos dados
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)
x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(x_norm, y, test_size = 0.3, random_state = 101)
x_train_norm_neural_network, x_test_norm_neural_network, y_train_norm_neural_network, y_test_norm_neural_network = train_test_split(x_norm, y_neural_network, test_size = 0.3, random_state = 101)
x_train_neural_network, x_test_neural_network, y_train_neural_network, y_test_neural_network = train_test_split(x, y_neural_network, test_size = 0.3, random_state = 101)


# # $\color{orange}{\textbf{Modelos de Machine Learning}}$

# # $\color{orange}{\textbf{Regressão logística}}$

# In[106]:


"""
Ajuste de hiperparâmetros para a regressão logística
"""
Logistic_regression = LogisticRegression()
penalty = ["l1", "l2", "elasticnet"]
C = np.array([0.0007, 0.001, 0.005, 0.009, 0.01])
param_grid = {"C": C, "penalty": penalty}
Grid_Logistic_regression = GridSearchCV(estimator = Logistic_regression, param_grid = param_grid, cv = 5, n_jobs=-1)
Grid_Logistic_regression.fit(x_train, y_train)
print(f"Penalty = {Grid_Logistic_regression.best_estimator_.penalty} // C = {Grid_Logistic_regression.best_estimator_.C}")


# In[107]:


"""
Processo de treinamento
"""
Logistic_regression = LogisticRegression(penalty = "l2", C = 0.005)
Logistic_regression.fit(x_train, y_train)


# In[108]:


"""
Processo de predição
"""
y_pred_Logistic_regression = Logistic_regression.predict(x_test)
"""
Report geral
"""
print(classification_report(y_test, y_pred_Logistic_regression))


# In[114]:


"""
Mapa de calor para a matriz de confusão
"""
font_heat_map = {"family": "serif", "weight": "normal", "size": 12, "color": "black"}
Matrix_Logistic_regression = confusion_matrix(y_test, y_pred_Logistic_regression)
sbn.heatmap(Matrix_Logistic_regression, 
            annot = True, 
            vmin = 0, 
            vmax = 140, 
            cmap = "Oranges", 
            fmt = "g", 
            linewidths=2, 
            linecolor="orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# In[115]:


"""
Predições de probabilidades para construção da curva roc
"""
y_pred_proba_Logistic_regression = Logistic_regression.predict_proba(x_test)
y_pred_proba_Logistic_regression = y_pred_proba_Logistic_regression[:, 1]
print(f"roc_auc_score_Logistic_regression = {roc_auc_score(y_test, y_pred_proba_Logistic_regression)}")


# # $\color{orange}{\textbf{KNN}}$

# In[116]:


KNN = KNeighborsClassifier()
n_neighbors = np.array([20, 10, 9, 8, 7, 6, 5, 4, 3])
p = np.array([1, 2, 3, 4, 5, 6, 7])
metric = ["euclidean", "manhattan", "minkowski", "chebyshev"]
param_grid = {"n_neighbors": n_neighbors, "p": p, "metric": metric}
Grid_KNN = GridSearchCV(estimator = KNN, param_grid = param_grid, cv = 5, n_jobs=-1)
Grid_KNN.fit(x_train_norm, y_train_norm)
print(f"n neighbors = {Grid_KNN.best_estimator_.n_neighbors} // p = {Grid_KNN.best_estimator_.p} // metric = {Grid_KNN.best_estimator_.metric}")


# In[117]:


KNN = KNeighborsClassifier(n_neighbors = 5, p = 1, metric = "euclidean")
KNN.fit(x_train_norm, y_train_norm)


# In[118]:


y_pred_KNN = KNN.predict(x_test_norm)
print(accuracy_score(y_test_norm, y_pred_KNN))


# In[119]:


print(classification_report(y_test_norm, y_pred_KNN))


# In[120]:


Matrix_KNN = confusion_matrix(y_test_norm, y_pred_KNN)
sbn.heatmap(Matrix_KNN, 
            annot = True, 
            vmin = 0, 
            vmax = 140, 
            cmap = "Oranges", 
            fmt = "g", 
            linewidths=2, 
            linecolor="orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# In[121]:


y_pred_proba_KNN = KNN.predict_proba(x_test_norm)
y_pred_proba_KNN = y_pred_proba_KNN[:, 1]
print(f"roc_auc_score_KNN = {roc_auc_score(y_test_norm, y_pred_proba_KNN)}")


# # $\color{orange}{\textbf{KMeans}}$

# In[122]:


K = pd.concat([x_norm, y], axis = 1)
Font2 = FontProperties(family = "serif",
                      weight = "bold",
                      style = "normal",
                      size = 12)
fig = plt.figure(figsize = (7, 10))
ax = fig.add_subplot(projection = "3d")
ax.scatter(K[K["target"] == 0]["Teff"], 
           K[K["target"] == 0]["logg"],
           K[K["target"] == 0]["[Fe/H]"], c = "darkred", label = "Gigantes vermelhas")
ax.scatter(K[K["target"] == 1]["Teff"], 
           K[K["target"] == 1]["logg"],
           K[K["target"] == 1]["[Fe/H]"], c = "orange", label = "Estrelas frias do tipo Sol")
ax.view_init(9, 10)
ax.set_xlabel("Teff $[K]$", fontdict = Font1)
ax.set_ylabel("Log(g) $[cm/s^{2}]$", fontdict = Font1)
ax.set_zlabel("[Fe/H] $[Sun]$", fontdict = Font1)
fig.patch.set_facecolor("white")
Cor_fundo = plt.gca()
Cor_fundo.set_facecolor("white")
Cor_fundo.patch.set_alpha(1)
plt.legend(frameon = False, prop = Font2, labelcolor = "gray")
plt.show()


# In[123]:


K_Means = KMeans(n_clusters=2, init = "k-means++", max_iter = 999, n_init = 10)
K_Means.fit(x_norm)


# In[125]:


y_predict_KMeans = K_Means.predict(x_norm)
y_predict_KMeans = pd.DataFrame(y_predict_KMeans)
y_predict_KMeans = y_predict_KMeans.replace({0:1, 1:0})
print(accuracy_score(y, y_predict_KMeans))


# In[126]:


print(classification_report(y, y_predict_KMeans))


# In[128]:


Matrix_KMeans = confusion_matrix(y, y_predict_KMeans)
sbn.heatmap(Matrix_KMeans, 
            annot = True, 
            vmin = 0, 
            vmax = 500, 
            cmap = "Oranges", 
            fmt = "g", 
            linewidths=2, 
            linecolor="orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# # $\color{orange}{\textbf{Naive Bayes}}$

# In[129]:


NB = GaussianNB()
NB.fit(x_train, y_train)


# In[130]:


y_pred_NB = NB.predict(x_test)
print(accuracy_score(y_test, y_pred_NB))


# In[131]:


print(classification_report(y_test, y_pred_NB))


# In[132]:


Matrix_NB = confusion_matrix(y_test, y_pred_NB)
sbn.heatmap(Matrix_NB, 
            annot = True, 
            vmin = 0, 
            vmax = 140, 
            cmap = "Oranges", 
            fmt = "g", 
            linewidths=2, 
            linecolor="orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# # $\color{orange}{\textbf{Random Forest}}$

# In[133]:


get_ipython().run_cell_magic('time', '', 'Random_Forest = RandomForestClassifier()\nmax_depth = np.array([1, 2, 3, 4, 5, 6, 7])\nmin_samples_split = np.array([2, 3, 4, 5])\nmin_samples_leaf = np.array([2, 3, 4, 5])\nparam_grid = {"max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf}\nGrid_Random_Forest = GridSearchCV(estimator = Random_Forest, param_grid = param_grid, cv = 5, n_jobs = -1)\nGrid_Random_Forest.fit(x_train, y_train)\nprint(f"max_depth = {Grid_Random_Forest.best_estimator_.max_depth} // min_samples_split = {Grid_Random_Forest.best_estimator_.min_samples_split} // min_samples_leaf = {Grid_Random_Forest.best_estimator_.min_samples_leaf}")')


# In[134]:


Random_Forest = RandomForestClassifier(max_depth = 2, min_samples_split = 2, min_samples_leaf = 2)
Random_Forest.fit(x_train, y_train)


# In[135]:


y_pred_Random_Forest = Random_Forest.predict(x_test)
print(accuracy_score(y_test, y_pred_Random_Forest))


# In[136]:


print(classification_report(y_test, y_pred_Random_Forest))


# In[137]:


Matrix_Random_Forest = confusion_matrix(y_test, y_pred_Random_Forest)
sbn.heatmap(Matrix_Random_Forest, 
            annot = True, 
            vmax = 150, 
            vmin = 0, 
            fmt = "g", 
            cmap = "Oranges", 
            linewidths = 2, 
            linecolor = "orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# # $\color{orange}{\textbf{Extra Trees}}$

# In[138]:


get_ipython().run_cell_magic('time', '', 'Extra_trees_classifier = ExtraTreesClassifier()\nmax_depth = np.array([1, 2, 3, 4, 5, 6])\nmin_samples_split = np.array([2, 3, 4, 5, 6, 7, 8, 9])\nmin_samples_leaf = np.array([2, 3, 4, 5])\nparam_grid = {"max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf}\nGrid_Extra_trees_classifier = GridSearchCV(estimator = Extra_trees_classifier, param_grid = param_grid, cv = 5, n_jobs = -1)\nGrid_Extra_trees_classifier.fit(x_train, y_train)\nprint(f"max_depth = {Grid_Extra_trees_classifier.best_estimator_.max_depth} // min_samples_split = {Grid_Extra_trees_classifier.best_estimator_.min_samples_split} // min_samples_leaf = {Grid_Extra_trees_classifier.best_estimator_.min_samples_leaf}")')


# In[139]:


Extra_trees_classifier = ExtraTreesClassifier(max_depth = 2, min_samples_split = 7, min_samples_leaf = 2)
Extra_trees_classifier.fit(x_train, y_train)


# In[140]:


y_pred_Extra_Trees = Extra_trees_classifier.predict(x_test)
print(accuracy_score(y_test, y_pred_Extra_Trees))


# In[141]:


print(classification_report(y_test, y_pred_Extra_Trees))


# In[142]:


Matrix_Extra_trees_classifier = confusion_matrix(y_test, y_pred_Extra_Trees)
sbn.heatmap(Matrix_Extra_trees_classifier, 
            annot = True, 
            vmax = 150, 
            vmin = 0, 
            fmt = "g", 
            cmap = "Oranges", 
            linewidths = 2, 
            linecolor = "orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# # $\color{orange}{\textbf{AdaBoost}}$

# In[143]:


Adaboost = AdaBoostClassifier(n_estimators=500)
learning_rate = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
param_grid = {"learning_rate": learning_rate}
Grid_Adaboost = GridSearchCV(estimator = Adaboost, param_grid = param_grid, cv = 5, n_jobs = -1)
Grid_Adaboost.fit(x_train, y_train)
print(f"learning_rate = {Grid_Adaboost.best_estimator_.learning_rate}")


# In[144]:


Adaboost = AdaBoostClassifier(n_estimators=500, learning_rate = 0.1)
Adaboost.fit(x_train, y_train)


# In[145]:


y_pred_Adaboost = Adaboost.predict(x_test)
print(accuracy_score(y_test, y_pred_Adaboost))


# In[146]:


print(classification_report(y_test, y_pred_Adaboost))


# In[147]:


Matrix_Adaboost = confusion_matrix(y_test, y_pred_Adaboost)
sbn.heatmap(Matrix_Adaboost, 
            annot = True, 
            vmax = 150, 
            vmin = 0, 
            fmt = "g", 
            cmap = "Oranges", 
            linewidths = 2, 
            linecolor = "orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# # $\color{orange}{\textbf{Gradient boosting}}$

# In[148]:


GradientBoosting = GradientBoostingClassifier(n_estimators=200)
learning_rate = np.array([0.1, 0.2, 0.3, 0.4])
min_samples_split = np.array([2, 3, 4, 5])
min_samples_leaf = np.array([2 , 3 ,4 ,5])
max_depth = np.array([2 , 3 ,4 ,5])
param_grid_GradientBoosting = {"learning_rate": learning_rate, "min_samples_split": min_samples_split, 
                              "min_samples_leaf": min_samples_leaf, "max_depth" : max_depth}
Grid_GradientBoosting = GridSearchCV(estimator = GradientBoosting, param_grid = param_grid_GradientBoosting, cv = 5, n_jobs=-1)
Grid_GradientBoosting.fit(x_train, y_train)
print(f"GradientBoosting: learning_rate  = {Grid_GradientBoosting.best_estimator_.learning_rate} // min_samples_split = {Grid_GradientBoosting.best_estimator_.min_samples_split} // min_samples_leaf = {Grid_GradientBoosting.best_estimator_.min_samples_leaf}")


# In[149]:


GradientBoosting = GradientBoostingClassifier(n_estimators=200, learning_rate  = 0.1, min_samples_split = 2, min_samples_leaf = 2)
GradientBoosting.fit(x_train, y_train)


# In[150]:


y_pred_GradientBoosting = GradientBoosting.predict(x_test)
print(accuracy_score(y_test, y_pred_GradientBoosting))


# In[151]:


print(classification_report(y_test, y_pred_GradientBoosting))


# In[152]:


Matrix_GradientBoosting = confusion_matrix(y_test, y_pred_GradientBoosting)
sbn.heatmap(Matrix_GradientBoosting, 
            annot = True, 
            vmax = 150, 
            vmin = 0, 
            fmt = "g", 
            cmap = "Oranges", 
            linewidths = 2, 
            linecolor = "orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# # $\color{orange}{\textbf{SVM}}$

# In[153]:


from sklearn.model_selection import KFold
from sklearn.svm import SVC
svc = SVC()
C = np.array([0.0001, 0.0006, 0.001, 0.005])
kernel = ["linear", "poly", "rbf", "sigmoid"]
degree = np.array([1, 2, 3, 4, 5, 6, 7, 8])
param_grid = {"C":C, "kernel": kernel, "degree": degree}
kfold = KFold(n_splits = 3, shuffle = True)
Grid_SVM = GridSearchCV(estimator = svc, param_grid = param_grid, cv = kfold, n_jobs=-1)
Grid_SVM.fit(x_train_norm, y_train_norm)
print(f"C = {Grid_SVM.best_estimator_.C} // kernel = {Grid_SVM.best_estimator_.kernel} // degree = {Grid_SVM.best_estimator_.degree}")


# In[154]:


svc = SVC(C = 0.005, kernel = "poly", degree = 1)
svc.fit(x_train_norm, y_train_norm)


# In[155]:


y_pred_SVC = svc.predict(x_test_norm)
print(accuracy_score(y_test_norm, y_pred_SVC))


# In[156]:


print(classification_report(y_test_norm, y_pred_SVC))


# In[157]:


Matrix_svm = confusion_matrix(y_test_norm, y_pred_SVC)
sbn.heatmap(Matrix_svm, 
            annot = True, 
            vmax = 150, 
            vmin = 0, 
            fmt = "g", 
            cmap = "Oranges", 
            linewidths = 2, 
            linecolor = "orange")
plt.ylabel("Valor real", fontdict = font_heat_map)
plt.xlabel("Valor predito", fontdict = font_heat_map)
plt.show()


# $\color{orange}{\textbf{Análise geral}}$

# In[158]:


Lista_de_modelos = ["Regressão logística", "KNN", "KMeans", 
                   "Naive Bayes", "Random Forest", "Extra Trees",
                   "Adaboost", "Gradient Boosting", "SVM"]
Lista_de_acuracias = [accuracy_score(y_test, y_pred_Logistic_regression),
                     accuracy_score(y_test_norm, y_pred_KNN),
                     accuracy_score(y, y_predict_KMeans),
                     accuracy_score(y_test, y_pred_NB),
                     accuracy_score(y_test, y_pred_Random_Forest),
                     accuracy_score(y_test, y_pred_Extra_Trees),
                     accuracy_score(y_test, y_pred_Adaboost),
                     accuracy_score(y_test, y_pred_GradientBoosting),
                     accuracy_score(y_test_norm, y_pred_SVC)]
Lista_de_modelos = pd.DataFrame(Lista_de_modelos, columns = ["Modelos"])
Lista_de_acuracias = pd.DataFrame(Lista_de_acuracias, columns = ["acc"])
acc_model = pd.concat([Lista_de_modelos, Lista_de_acuracias], axis = 1)
acc_model


# In[159]:


font5 = {"family": "serif", "weight": "bold", "color": "gray", "size": 12.4}
fig, ax = plt.subplots(figsize = (10, 7))
sbn.barplot(x = "acc", y = "Modelos", data = acc_model, color = "orange")
plt.xlabel("Acurácia",fontdict = font5)
plt.ylabel("Modelos", fontdict = font5)
for axis in ["left", "right", "top", "bottom"]:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color("gray")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis = "both", direction = "in", labelcolor = "gray", labelsize = 12.4)
ax.tick_params(which = "minor", direction = "in", width = 2, color = "gray", left = False)
ax.tick_params(which = "major", direction = "in", color = "gray", length=3.4, width = 2)
fig.patch.set_facecolor("white")
Cor_fundo = plt.gca()
Cor_fundo.set_facecolor("white")
Cor_fundo.patch.set_alpha(1)
plt.xlim(0.8, 1)
plt.show()


# # $\color{orange}{\textbf{Redes neurais}}$

# In[160]:


"""
Montagem da rede
"""
Modelo = Sequential()
Modelo.add(Dense(4, input_dim = 3, kernel_initializer = "normal", activation = "relu"))
Modelo.add(Dense(2, kernel_initializer = "normal", activation = "softmax"))


# In[161]:


from keras.optimizers import Adam
optimizer = Adam() # Optimizador Adam()
Modelo.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["acc"])
History = Modelo.fit(x_train_norm_neural_network, y_train_norm_neural_network, epochs = 500, batch_size = 200, validation_data=(x_test_norm_neural_network, y_test_norm_neural_network), verbose = 0)


# In[162]:


acc_test = History.history["val_acc"]
max(acc_test)


# In[163]:


"""
Gráfico de comparação entre acurácia de treino e teste
"""
acc_train = History.history["acc"]
epochs = range(1, len(acc_train) + 1)
fig, ax = plt.subplots(figsize = (9, 7))
ax.plot(epochs, acc_train, "--g", color = "darkred", label = "Acurácia treino")
ax.plot(epochs, acc_test, "-b", color = "orange", label = "Acurácia teste")
for axis in ["left", "right", "top", "bottom"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("gray")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis = "both", direction = "in", labelcolor = "gray", labelsize = 13, top = True, right = True, left = True, bottom = True)
ax.tick_params(which='minor', direction = "in", length=2, color='gray', width = 2, top = True, right = True, left = True, bottom = True)
ax.tick_params(which='major', direction = "in", color='gray', length=3.4, width = 2, top = True, right = True, left = True, bottom = True)
ax.legend(frameon = False, prop = Font2, labelcolor = "gray")
ax.set_xlabel("Epochs", fontdict = font5)
ax.set_ylabel("Acurácia", fontdict = font5)
fig.patch.set_facecolor("white")
Cor_fundo = plt.gca()
Cor_fundo.set_facecolor("white")
Cor_fundo.patch.set_alpha(1)
plt.show()

