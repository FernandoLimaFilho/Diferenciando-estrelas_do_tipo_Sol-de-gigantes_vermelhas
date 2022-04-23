#!/usr/bin/env python
# coding: utf-8

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

