#!/usr/bin/env python
# coding: utf-8

# # ANÁLISE 003 - POR PAÍS

# ## PRIMEIRA BASE DE DADOS : MOBILIDADE - GOOGLE

# BASE GOOGLE  
# Da mesma data que extraimos os dados para e estado de Sào Paulo, iremos extrair para BRASIL, CHILE e REINO UNIDO. Os seguintes tratamentos foram feitos ainda no Excel para acelerar a análise:
# - Selecionar somente as linhas com os dados consolidados do país, ou sejam, com sub_region1 (estados/provincias) e sub_region2 (municípios) vazios.  
# - Colunas: manter somente country_region (para que possamos juntar os paises analisados numa só base) e de date até o final.  
# - Juntar as 6 planilhas excel numa só.

# #### ATENÇÃO - a escolha do Reino Unido teve como referência o trabalho de [BASELLINI et al, 2019], que também utilizou os dados do Google Mobility Reports e, embora tenha abrangido um escopo geográfico um pouco mais restrito (apenas Inglaterra e País de Gales) mostrou um cenário bastante detalhado das medidas pelo governo britânico, no sentido de controlar o avanço da doença, em especial os índices de mortalidade da mesma. 

# In[98]:


#Base resultante : BR_CL_GB_Mobility_Report.xlsx
#vamos renomear as colunas para o português conforme feito na análise para SP


# In[99]:


import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from scipy.stats.stats import pearsonr
from datetime import datetime  

# Ignorar warnings não prejudiciais
import warnings
warnings.filterwarnings("ignore")


# ATENÇÃO, essa planilha foi montada à mão e contém os dados originais, não normalizados !

# In[100]:


dados_mob = pd.read_excel('BR_CL_GB_Mobility_Report.xlsx')


# Para esta massa de dados, mais adiante, limitaremos a análise ao ano de 2020, já que os procedimentos de vacinação no Reino Unido começaram ainda em dezembro (https://g1.globo.com/bemestar/vacina/noticia/2020/12/07/reino-unido-anuncia-que-vacinacao-contra-covid-19-comeca-nesta-terca-8.ghtml) e portanto já no primeiro semestre de 2021 o efeito de tais procedimentos começou a influenciar a curva de casos.

# In[101]:


dados_mob.columns = ['pais', 'data', 'comercio_recreacao', 'alimentacao_farmacia', 
                'parques', 'estacoes_transporte', 'locais_trabalho', 'residencias']


# In[102]:


#verificar valores ausentes nas colunas - não há lacunas a preencher
dados_mob.isna().sum()


# In[103]:


def filtroDataFinal(df,d):
    filtro_data = (df['data'] <= d)
    dados_data = df[filtro_data]
    return(dados_data)


# In[104]:


#já faz aqui a restrição de data (31/12/2020)
dados_mob = filtroDataFinal(dados_mob,'2020-12-31')


# Temos agora, então, uma base em excel contendo o período fevereiro a dezembro/2020 para a variação das categorias de mobilidade para Brasil, Chile e Grã-Bretanha.

# ## SEGUNDA BASE DE DADOS : OCORRÊNCIAS - OWID

# Fonte de dados para casos - Brasil, Chile e Gra-Bretanha - extraido da base do site OUR WORLD IN DATA (OWID - Oxford) em 09/09/2021 - https://ourworldindata.org/explorers/coronavirus-data-explorer?zoomToSelection=true&time=2020-03-01..latest&facet=none&pickerSort=asc&pickerMetric=location&Metric=Confirmed+cases&Interval=7-day+rolling+average&Relative+to+Population=true&Align+outbreaks=false&country=BRA~CHL~GBR

# A formatação da base foi feita diretamente no excel, e consistiu em : (1) exclusão das colunas que não seriam utilizadas; (2) exclusão das linhas fora do escopo desta análise (Brasil, Reino Unido e Chile); (3) inclusão da coluna da data aproximada de contaminação (utilizando o tempo médio entre contaminação e notificação de 15 dias para o Brasil, 10 dias para o Chile e 3 dias para Reino Unido).

# In[105]:


dados_casos = pd.read_excel('owid-covid-data-BR-CL-UK.xlsx')
dados_casos


# ## PRÓXIMA ETAPA: fazer o merge das bases de mobilidade e de casos acima,  amarrando por país e por data - na base de casos, deve-se utilizar a data da contaminação (aproximada)

# In[106]:


dados_casos.columns


# In[107]:


dados_mob.columns


# In[108]:


dados_casos.columns = ['pais','data_notif','data','casos novos','obitos novos','vacinacoes novas']


# In[109]:


def converteTimestamp(df,col_data):

    lista = []

    for dt in df[col_data]:
        dti = datetime.strptime(str(dt), '%Y-%m-%d %H:%M:%S')
        lista.append(dti.strftime('%Y-%m-%d'))

    df[col_data] = lista
    
    return(df)


# In[110]:


dados_casos = converteTimestamp(dados_casos,'data')


# ### JOIN

# In[111]:


dados_mob_casos_BR_CL_UK = pd.merge(dados_mob, dados_casos, on=["pais","data"], how="inner")


# In[112]:


#elimina a coluna dados_notif que nao tera utiilidade no momento
dados_mob_casos_BR_CL_UK = dados_mob_casos_BR_CL_UK.drop('data_notif', axis=1)


# In[113]:


#preenche os valores faltantes usando o modelo MissForest (random forest)

dados_mob_casos_BR_CL_UK.isna().sum()


# In[114]:


def preencheNan(df,cols):
    
    import sklearn.neighbors._base
    import sys

    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

    from missingpy import MissForest
    imputer = MissForest()

    X = df[cols]

    X_imputed = imputer.fit_transform(X)

    df[cols] = X_imputed
    
    return(df)


# In[115]:


novas_cols = ['obitos novos','vacinacoes novas']
dados_mob_casos_BR_CL_UK = preencheNan(dados_mob_casos_BR_CL_UK,novas_cols)


# In[116]:


dados_mob_casos_BR_CL_UK.isna().sum()


# In[117]:


def normaliza_minmax(df,cols):
    for x in df[cols]:
        df[x] = (df[x] - df[x].min()) / (df[x].max() - df[x].min())
    return(df[cols])


# In[118]:


#inclusão da coluna de variação média de mobilidade, pegando a média das 6 colunas de
#variação
dados_mob_casos_BR_CL_UK['med_var_mob'] =  (dados_mob_casos_BR_CL_UK['comercio_recreacao'] +
                                     dados_mob_casos_BR_CL_UK['alimentacao_farmacia'] +
                                     dados_mob_casos_BR_CL_UK['parques'] +
                                     dados_mob_casos_BR_CL_UK['estacoes_transporte'] +
                                     dados_mob_casos_BR_CL_UK['locais_trabalho'] +
                                     dados_mob_casos_BR_CL_UK['residencias']) / 6


# In[120]:


#acresccentar a coluna com a média móvel (7 dias) para os casos novos e outra para a mobilidade. 
#Veremos mais dessa média nos cálculos envolvendo tempo
dados_mob_casos_BR_CL_UK['MMS7-casos'] = dados_mob_casos_BR_CL_UK['casos novos'].rolling(window=7).mean()
dados_mob_casos_BR_CL_UK['MMS7-mob']   = dados_mob_casos_BR_CL_UK['med_var_mob'].rolling(window=7).mean()


# In[121]:


#normaliza as colunas de valores

colunas = ['comercio_recreacao','alimentacao_farmacia','parques','estacoes_transporte',
           'locais_trabalho','residencias','casos novos', 'obitos novos', 'vacinacoes novas',
           'med_var_mob','MMS7-casos','MMS7-mob']

dados_mob_casos_BR_CL_UK[colunas] = normaliza_minmax(dados_mob_casos_BR_CL_UK,colunas)


# In[122]:


#Aqui já podemos substituir "United Kingdom" por "Reino Unido" e "Brazil" por "Brasil" 
#para manter a padronização do trabalho
dados_mob_casos_BR_CL_UK.loc[dados_mob_casos_BR_CL_UK.pais == 'United Kingdom','pais']='Reino Unido'
dados_mob_casos_BR_CL_UK.loc[dados_mob_casos_BR_CL_UK.pais == 'Brazil','pais']='Brasil'


# In[123]:


#preenche os valores faltantes
dados_mob_casos_BR_CL_UK.isna().sum()


# In[125]:


#para evitar valores distorcidos no início, vamos considerar somente os dados a partir de 01/04/20
dados_mob_casos_BR_CL_UK = dados_mob_casos_BR_CL_UK[dados_mob_casos_BR_CL_UK['data'] > '2020-03-31']


# In[126]:


dados_mob_casos_BR_CL_UK.isna().sum()


# In[128]:


#para analisar no tempo, temos que fixar o local. Fixemos inicialmente município = SAO PAULO
def dados_pais(pais):
    filtro_pais =(dados_mob_casos_BR_CL_UK['pais'] == pais)
    dados = dados_mob_casos_BR_CL_UK[filtro_pais]
    return(dados)


# In[129]:


def correlacao_maxima(pais):
    
    df = dados_pais(pais)
    df = df[colunas] #mantem apenas as colunas que devem entrar na correlação
    rs = np.random.RandomState(0)
    corr = df.corr()['casos novos'] #obtem a correlação de cada categoria com casos novos
    cat = corr[:-1].max() #retorna o valor da maior correlação positiva
    return(cat)


# In[130]:


def categoria_correlacao_maxima(pais):
    
    df = dados_pais(pais)
    df = df[colunas] #mantem apenas as colunas que devem entrar na correlação
    rs = np.random.RandomState(0)
    corr = df.corr()['casos novos'] #obtem a correlação de cada categoria com casos novos
    ind = corr[:-1].argmax() #retorna o nome da categoria de maior correlação positiva
    return(colunas[ind])


# In[131]:


import time

#PREENCHIMENTO DAS COLUNAS DE CORRELAÇÃO - AQUI RODA RÁPIDO, menos de 3 segundos
ini = time.time() 
lista_cat = []
lista_corr = []
perc = 0
total_linhas = dados_mob_casos_BR_CL_UK.shape[0]
for p in dados_mob_casos_BR_CL_UK['pais']:
    lista_cat.append(categoria_correlacao_maxima(p))
    lista_corr.append(correlacao_maxima(p))
    perc += (1/total_linhas)*100
    print('*** processado ',perc,'%')
    
dados_mob_casos_BR_CL_UK['CAT_MOB_PRINCIPAL'] = lista_cat
dados_mob_casos_BR_CL_UK['CORR_CAT_MOB_PRINCIPAL'] = lista_corr    
fim = time.time()   
print("********* processou em ",fim-ini," segundos")
   


# # Geração dos gráficos da evolução no tempo e correlações para os países em foco

# In[132]:


def filtroPais(df,p):
    filtro_pais =(df['pais'] == p)
    dados_pais = df[filtro_pais]
    return(dados_pais)


# In[133]:


def datasToInt(s):
    serieInt = []
    i = 0
    for d in s:
        serieInt.append(i)
        i += 1
        
    return(serieInt)    


# In[134]:


def plotPais(p):
    
    # Filtra o pais
    serie = filtroPais(dados_mob_casos_BR_CL_UK, p) 
    
    linha = "*"* 126
    print(linha)
    print("ANÁLISE DE MOBILIDADE X NOVOS CASOS DE INFECÇÃO POR COVID - 19 PARA O : ",p.upper(),
          " - período de ",serie['data'].min()," a ",serie['data'].max(),"\n(Fontes : GOOGLE MOBILITY REPORT + OUR WORLD IN DATA - OXFORD)")
    print(linha)
    
    # Normaliza os dados --- NÃO, JA FOI NORMALIZADO
    #colunas_tratadas = ['MMS7-casos','MMS7-mob']
    #serie[colunas_tratadas] = normaliza_minmax(serie,colunas_tratadas)
    
    #----- DESENHA O GRÁFICO DE LINHAS --------------------------------------------------------
    
    fig = plt.figure(figsize=(35,15))
    plt.plot(serie['data'], serie['MMS7-mob'], color = 'blue', label = 'MM7 MOBILIDADE',linewidth=7)
    plt.plot(serie['data'], serie['MMS7-casos'],color = 'red', label = 'MM7 CASOS DATA OCORRÊNCIA REAL',linewidth=7)
    #plt.plot(serie['data_orig'], serie['MMS7-casos'],color = 'orange', label = 'MM7 CASOS DATA COMUNICADO OFICIAL',linewidth=7,linestyle=':')
    
    plt.legend(loc=4,fontsize='xx-large')
    plt.title("EVOLUÇÃO NO TEMPO DE MOBILIDADE X NOVOS CASOS PARA O "+ p.upper(),fontsize='xx-large')
    
    serie_eixo = list(serie['data'])
    i = 0
    for i in range(len(serie_eixo)):
        if (i % 10) != 0:
            serie_eixo[i] = ''
    plt.xticks(rotation=90)    
    plt.xticks(serie_eixo)
    
    plt.show()
    
    #----- CALCULA A CORRELAÇÃO ---------------------------------------------------------------
    
    mob_col = serie['MMS7-mob'].values
    casos_col = serie['MMS7-casos'].values
    corr , _ = pearsonr(mob_col, casos_col)
    
    linha = "-"* 127
    print(linha)
    print("CORRELAÇÃO LINEAR ENTRE AS MÉDIAS MÓVEIS DE VARIAÇÃO DE MOBILIDADE E DE CASOS NOVOS PARA O ",
          p.upper(), "\nPeríodo de",serie['data'].min(),"a", serie['data'].max(),":",corr)
    print(linha)
    
    #----- DESENHA O PLOT DE CORRELAÇÃO -------------------------------------------------------
    
    print("\n\n>>> A escala de cores do gráfico corresponde ao tempo (valores maiores == datas mais recentes) <<<\n\n")
    categ = serie['CAT_MOB_PRINCIPAL'].unique()
    
    fig = plt.figure(figsize=(35,15))
    plt.xlabel("MMS7 MOBILIDADE",fontsize='xx-large')
    plt.ylabel("MMS7 CASOS",fontsize='xx-large')
    plt.scatter(serie['MMS7-mob'], serie['MMS7-casos'],
                s=2500,alpha=0.5,edgecolors='face', c=datasToInt(serie['data']))
    plt.title("CORRELAÇÃO VARIAÇÃO DE MOBILIDADE X NOVOS CASOS PARA " 
              + m + " (categoria principal de mobilidade : "+ categ + ")",fontsize='xx-large')
    
    z = np.polyfit(serie['MMS7-mob'], serie['MMS7-casos'], 1)
    p = np.poly1d(z)
    plt.plot(serie['MMS7-mob'],p(serie['MMS7-mob']),"r:",linewidth=8)
    plt.colorbar() 
    plt.show()
    
    return


# In[135]:


def matrizCorrelacoes(p):
    
    df = filtroPais(dados_mob_casos_BR_CL_UK,p)
    df = df[colunas] #mantem apenas as colunas que devem entrar na correlação
    
    linha = "*"* 126
    print(linha)
    print("MAPA DE CORRELAÇÕES ENTRE AS CATEGORIAS DE VARIAÇÃO DE MOBILIDADE (RELATÓRIO GOOGLE) X NOVOS CASOS DE INFECÇÃO POR COVID - 19 (BASE OWID - OXFORD) NO ",p.upper(), "- período de",dados_mob_casos_BR_CL_UK['data'].min(),"a", dados_mob_casos_BR_CL_UK['data'].max())
    print(linha)
    
    #-----HEATMAP
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,annot=True)
    
    #-----CORRELATIONS
    g = sns.pairplot(df,diag_kind="kde",corner=True)
    g.map_lower(sns.kdeplot, levels=4, color=".2")
    plt.figure(figsize=(20, 9))
    
    return  


# In[136]:


plotPais('Brasil')


# In[138]:


colunas = colunas[:-5]


# In[139]:


matrizCorrelacoes('Brasil')


# In[140]:


plotPais('Chile')


# In[141]:


matrizCorrelacoes('Chile')


# In[142]:


plotPais('Reino Unido')


# In[143]:


matrizCorrelacoes('Reino Unido')


# In[ ]:





# In[ ]:





# In[ ]:




