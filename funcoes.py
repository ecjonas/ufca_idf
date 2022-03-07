# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PyQt5.QtWidgets import QFileDialog

from scipy import integrate
from types import SimpleNamespace
from scipy import stats
import csv
import math
import numpy as np
import pandas as pd

#MÉTODO 01: LER O ARQUIVO TXT E RETORNAR A MATRIZ HID [ANO] [PRECIPITAÇÃO]    

#FIM DA MÉTODO 01

def filestring():
    fname = QFileDialog.getOpenFileName(None, 'Selecionar Arquivo', 
   '',"txt (*.txt)")
    f = open(str(fname[0]),'r')
    texto = f.readlines()
    dados = []
    for line in texto:
        dados.append(line)
    dados = list(csv.reader((line.replace(',', ';') for line in dados), delimiter= ';'))
    informacao = []
    for i in range(8):
        informacao.append(dados[0])
        dados = np.delete(dados,(0),axis=0)
    return dados, informacao 

def save(precipitacao):   
    fileName = QFileDialog.getSaveFileName(None, 'Salvar Como...', '','txt (*.txt)') 
    arquivo = str(fileName[0])
    with open(arquivo, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(precipitacao)

def lerArquivo(arquivo):
    f = open(arquivo, 'r')
    texto = f.readlines()
    dados = []
    for line in texto:
        dados.append(line)
    dados = list(csv.reader((line.replace(',', ';') for line in dados), delimiter= ';'))
    informacao = []
    for i in range(8):
        informacao.append(dados[0])
        dados = np.delete(dados,(0),axis=0)
    return dados, informacao
        
def dadosPostos(arquivo):
    f = open(arquivo, encoding='utf8')
    texto = f.readlines()
    dados = []
    for line in texto:
        dados.append(line)
    dados = list(csv.reader((line.replace(',', ';') for line in dados), delimiter= ';'))
    dados = np.delete(dados,(0),axis=0)
    dadosArquivo = pd.DataFrame(data=dados, columns=['UF', 'Codigo', 'Cidade', 'Nome_Posto', 'Latitude', 'Longitude'])
    estado = dadosArquivo['UF'].drop_duplicates()
    estado = estado.values
    return dadosArquivo, estado

def tabelaDistricuicaoFrequencia(precipitacao):
    #Distribuição de Frequência
    maximaSeries = pd.Series(precipitacao)
    maximasOrdem = precipitacao
    maximasOrdem.sort()
    numClasses = int(math.sqrt(len(precipitacao)))
    amplitudeTotal = max(precipitacao) - min(precipitacao)
    intervaloClasse = amplitudeTotal / numClasses
    limiteInferior = min(precipitacao)
    frequenciaAcumulada = 0
    tabelaFrequencia = np.zeros((7,numClasses))
    for i in range(numClasses):
        limiteSuperior = limiteInferior + intervaloClasse
        frequencia = 0
        
        for j in range(len(precipitacao)):
            valor = maximasOrdem[j]
            if limiteInferior<=valor<limiteSuperior:
                frequencia += 1
        porcFrequencia = (frequencia / len(precipitacao)) * 100
        frequenciaAcumulada += frequencia
        porcFrequenciaAcumulada = (frequenciaAcumulada / len(precipitacao)) * 100    
        pontoMedio = (limiteInferior + limiteSuperior) / 2
        tabelaFrequencia [0,i] = round(limiteInferior,2)
        tabelaFrequencia [1,i] = round(limiteSuperior,2)
        tabelaFrequencia [2,i] = int(frequencia)
        tabelaFrequencia [3,i] = round(porcFrequencia,2)
        tabelaFrequencia [4,i] = int(frequenciaAcumulada)
        tabelaFrequencia [5,i] = round(porcFrequenciaAcumulada,2)
        tabelaFrequencia [6,i] = round(pontoMedio,2)
        limiteInferior = limiteSuperior
    
    return numClasses, maximaSeries, amplitudeTotal, intervaloClasse, tabelaFrequencia

def gumbel(maximas):
    beta = np.mean(maximas) - 0.451*np.std(maximas)
    alpha = np.sqrt(6/(math.pow(np.pi,2)))*np.std(maximas)
    funGumbel = lambda x: (1/alpha)*math.exp(-((x-beta)/alpha)-(math.exp(-((x-beta)/alpha))))
    return funGumbel

def inverseGumbel(maximas):
    beta = np.mean(maximas) - 0.451*np.std(maximas)
    alpha = np.sqrt(6/(math.pow(np.pi,2)))*np.std(maximas)
    invGumbel = lambda x: beta - alpha*np.log(-np.log(1-(1/x)))
    return invGumbel

def distribuicaoFrequencia(maximas):
    maximasOrdem = maximas
    maximasOrdem.sort()
    numClasses = int(math.sqrt(len(maximas)))
    amplitudeTotal = max(maximas) - min(maximas)
    intervaloClasse = round(amplitudeTotal / numClasses,3)
    limiteInferior = min(maximas)
    return SimpleNamespace(num = numClasses,amp = amplitudeTotal,inter = intervaloClasse,limInf = limiteInferior,ma = maximasOrdem)

#Parêmetros das distribuições
def parametrosDist(maximas):
    LogMaximas = []
    for i in range(0,len(maximas),1):
        LogMaximas.append(math.log(maximas[i]))
        
    muy = np.mean(LogMaximas)
    sigmay = np.std(LogMaximas)
    A0 = np.mean(maximas) - np.exp(muy+(0.5*(sigmay**2)))
    
    alphaII = math.pow(np.mean(maximas)/np.std(maximas),2)
    betaII = math.pow(np.std(maximas),2)/np.mean(maximas)
    asx = np.std(maximas)/np.mean(maximas)
    alphaIII = 4/(asx**2)
    betaIII = (np.std(maximas)*asx)/2
    gama = np.mean(maximas) - ((2*np.std(maximas))/asx) 
    
    return muy, sigmay, A0, alphaII, betaII, alphaIII, betaIII, gama


#Teste Quiquadrado
def testeQuiQuadrado(numDist, maximas):
    #Parâmetros das distribuições
    muy, sigmay, A0, alphaII, betaII, alphaIII, betaIII, gama = parametrosDist(maximas)
    
    valores = distribuicaoFrequencia(maximas)
    numClasses = valores.num
    intervaloClasse = valores.inter
    limiteInferior = valores.limInf
    maximasOrdem = valores.ma
    matrizQui = np.zeros((numClasses,6))
    for i in range(numClasses):
        limiteSuperior = limiteInferior + intervaloClasse
        frequencia = 0
        
        for j in range(len(maximas)):
            valor = maximasOrdem[j]
            if limiteInferior<=valor<limiteSuperior:
                frequencia += 1
    
        matrizQui [i][0] = round(limiteInferior,2)
        matrizQui [i][1] = round(limiteSuperior,2)
        matrizQui [i][2] = int(frequencia)
        if numDist == 1:
            dist = stats.gamma(a = alphaII,scale=betaII)
                
        if numDist == 2:
            dist = stats.gamma(a = alphaIII,loc=gama,scale=betaIII)
         
        if numDist == 3:
            dist = stats.lognorm(s=sigmay, scale=math.exp(muy))
                
        if numDist == 4:
            dist = stats.lognorm(s=sigmay, loc=A0, scale=math.exp(muy))
            
        if numDist<5:
            if i==0:
                fi = dist.cdf(matrizQui [i][1]) - dist.cdf(0)
                
            elif i==(numClasses-1):
                fi = dist.cdf(np.inf) - dist.cdf(matrizQui [i][0])
                
            else:
                fi = dist.cdf(matrizQui [i][1]) - dist.cdf(matrizQui [i][0])
         
        if numDist == 5:
            if i==0:
                fi, err = integrate.quad(gumbel(maximas),0,matrizQui [i][1])
                
            elif i==(numClasses-1):
                fi, err = integrate.quad(gumbel(maximas),matrizQui [i][0],np.inf)
                
            else:
                fi, err = integrate.quad(gumbel(maximas),matrizQui [i][0],matrizQui [i][1])
    
        matrizQui [i][3] = fi
        matrizQui [i][4] = matrizQui [i][3] * len(maximas)
        matrizQui [i][5] = ((math.pow(matrizQui [i][2] - matrizQui [i][4],2))) / matrizQui [i][4]
        limiteInferior = limiteSuperior
    matrizQui [numClasses-1][2] = matrizQui [numClasses-1][2] + 1
    matrizQui [numClasses-1][5] = ((math.pow((matrizQui [numClasses-1][2] - matrizQui [numClasses-1][4]),2))) / matrizQui [i][4]
    
    return numClasses, matrizQui

#Indice de Comparação
def indiceComparacao(numDist, maximas):
    maximas.sort()
    #Parâmetros das distribuições
    muy, sigmay, A0, alphaII, betaII, alphaIII, betaIII, gama = parametrosDist(maximas)    
    li = len(maximas)
    matrizIC = np.zeros((li,11))
    somaObserv = 0
    somaXY = 0
    somaX2 = 0
    somaDQM = 0
    somaDQR = 0
    somaDPMA = 0
    for i in range(0,li):
        #Primeira coluna
        matrizIC [i][0] = maximas [i]
        #Segunda coluna
        matrizIC [i][1] = li - i
        if numDist == 1:
            #Terceira coluna
            matrizIC [i][2] = (matrizIC [i][1] - 0.5) / li
            #Quarta coluna
            dist = stats.gamma(a = alphaII,scale=betaII)
            fi = dist.cdf(matrizIC [i][0])
        if numDist == 2:
            #Terceira coluna
            matrizIC [i][2] = (matrizIC [i][1] - 0.5) / li
            #Quarta coluna
            dist = stats.gamma(a = alphaIII,loc=gama,scale=betaIII)
            fi = dist.cdf(matrizIC [i][0])
        if numDist == 3:
            #Terceira coluna
            matrizIC [i][2] = (matrizIC [i][1] - (3 / 8)) / (li + 0.25)
            #Quarta coluna
            dist = stats.lognorm(s=sigmay, scale=math.exp(muy))
            fi = dist.cdf(matrizIC [i][0])
        if numDist == 4:
            #Terceira coluna
            matrizIC [i][2] = (matrizIC [i][1] - (3 / 8)) / (li + 0.25)
            #Quarta coluna
            dist = stats.lognorm(s=sigmay, loc=A0, scale=math.exp(muy))
            fi = dist.cdf(matrizIC [i][0])
        if numDist == 5:
            #Terceira coluna
            matrizIC [i][2] = (matrizIC [i][1] - 0.44) / (li + 0.12)
            #Quarta coluna
            fi, err =  integrate.quad(gumbel(maximas),0, matrizIC [i][0])
            
        matrizIC [i][3] = 1 - fi
        #Quinta coluna
        matrizIC [i][4] = math.pow(((matrizIC [i][2] - matrizIC [i][3]) / matrizIC [i][3]),2)
        #Sexta coluna
        matrizIC [i][5] = math.pow((matrizIC[i][2] - matrizIC[i][3]),2)
        #Setima coluna
        matrizIC [i][6] = ((abs(matrizIC[i][2] - matrizIC[i][3])) * 100) / matrizIC[i][2]
        #Oitava coluna
        matrizIC [i][7] = matrizIC [i][2] * matrizIC [i][3]
        #Nona coluna
        matrizIC [i][8] = math.pow(matrizIC [i][3],2)
        
        #Cálculo da soma dos valores observados
        somaObserv += matrizIC [i][2]
        #Cálculo da soma da multiplicação dos valores calculados pelos observados
        somaXY += matrizIC [i][7] 
        #Cálculo da soma dos valores calculados ao quadrado
        somaX2 += matrizIC [i][8]
        #Cálculo do somatório dos desvios DQM,DQR e DPMA
        somaDQM += matrizIC [i][4]
        somaDQR += matrizIC [i][5]
        somaDPMA += matrizIC [i][6]
    
    #Cálculo da média dos valores observados
    medObserv = somaObserv / li
    #Cálculo do coeficiente ângular da reta de tendência
    coefAng = somaXY / somaX2
    #Cálculo do Desvio Quadrático Médio
    DQM = math.sqrt(somaDQM / li)
    #Cálculo do Desvio Quadrático Residual
    DQR = math.sqrt(somaDQR / (li - 1))
    #Cáculo do Desvio Percentual Médio Absoluto
    DPMA = somaDPMA / li
    
    somaQR = 0
    somaQTo = 0
    for i in range(0,li):
        #Décima coluna
        matrizIC [i][9] = math.pow(matrizIC [i][2] - (matrizIC [i][3] * coefAng),2)
        #Décima primeira coluna
        matrizIC [i][10] = math.pow(math.pow(matrizIC [i][2],2) - medObserv,2)
        #Cálculo da soma da diferença quadrática dos valores observados pelos projetados
        somaQR += matrizIC [i][9]
        #Cáculo da soma dos valores observados ao quadrado - média 
        somaQTo += matrizIC [i][10]
    
    coefR2 = 1 - (somaQR / somaQTo)
    return DQM, DQR, DPMA, coefAng, coefR2, matrizIC


#Teste Kolmogorov-Smirnov

y = np.array([104.3, 97.9, 89.2, 92.7, 98, 141.7, 81.1, 97.3, 72, 93.9, 83.8, 122.8, 87.6, 101, 97.8, 59.9, 49.4, 57, 68.2, 83.2, 60.6, 50.1, 68.7, 117.1, 80.2, 43.6, 66.8, 118.4, 110.4, 99.1, 71.6, 62.6, 61.2, 46.8, 79, 96.3, 77.6, 69.3, 67.2, 72.4, 78, 141.8, 100.7, 87.4, 100.2, 166.9, 74.8, 133.4, 85.1, 78.9, 76.4, 64.2, 53.1, 112.2, 110.8, 82.2, 88.1, 80.9, 89.8, 114.9, 63.6, 57.3])

# Calculating the required values

#Verifica o valor crítico do Teste Kolmogorov-Smirnov
def kolmogorov_smirnov_critico(maximas):
    # table of critical values for the kolmogorov-smirnov test - 95% confidence
    # Source: https://www.soest.hawaii.edu/GG/FACULTY/ITO/GG413/K_S_Table_one_Sample.pdf
    # Source: http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
    # alpha = 0.05 (95% confidential level)
    n = len(maximas)
    if n <= 40:
        # valores entre 1 e 40
        kolmogorov_critico = [0.97500, 0.84189, 0.70760, 0.62394, 0.56328, 0.51926, 0.48342, 0.45427, 0.43001, 0.40925, 
                      0.39122, 0.37543, 0.36143, 0.34890, 0.33760, 0.32733, 0.31796, 0.30936, 0.30143, 0.29408, 
                      0.28724, 0.28087, 0.27490, 0.26931, 0.26404, 0.25907, 0.25438, 0.24993, 0.24571, 0.24170, 
                      0.23788, 0.23424, 0.23076, 0.22743, 0.22425, 0.22119, 0.21826, 0.21544, 0.21273, 0.21012]
        ks_crit = kolmogorov_critico[n - 1]
    elif n > 40:
        # valores acima de 40:
        kolmogorov_critico = 1.36/(np.sqrt(n))
        ks_crit = kolmogorov_critico
    else:
        pass            
            
    return ks_crit

def kolmogorov_smirnov_calculado(numDist, maximas):
    maximas.sort()
    #Parâmetros das distribuições
    muy, sigmay, A0, alphaII, betaII, alphaIII, betaIII, gama = parametrosDist(maximas)
    matrizKS = np.zeros((len(maximas),7))   
    for i in range(len(maximas)):
        #Primeira coluna
        matrizKS [i][0] = maximas [i]
        #Segunda coluna
        matrizKS [i][1] = i+1
        #Terceira coluna
        matrizKS [i][2] = (i+1)/len(maximas)
        #Quarta coluna
        matrizKS [i][3] = i/len(maximas)
        #Quinta coluna
        if numDist == 1:
            dist = stats.gamma(a = alphaII,scale=betaII)
            fi = dist.cdf(matrizKS [i][0])
        if numDist == 2:
            dist = stats.gamma(a = alphaIII,loc=gama,scale=betaIII)
            fi = dist.cdf(matrizKS [i][0])
        if numDist == 3:
            dist = stats.lognorm(s=sigmay, scale=math.exp(muy))
            fi = dist.cdf(matrizKS [i][0])
        if numDist == 4:
            dist = stats.lognorm(s=sigmay, loc=A0, scale=math.exp(muy))
            fi = dist.cdf(matrizKS [i][0])
        if numDist == 5:
            fi, err =  integrate.quad(gumbel(maximas),0, matrizKS [i][0])
        matrizKS [i][4] = fi
        #Sexta coluna
        matrizKS [i][5] = abs(matrizKS[i][4] - matrizKS [i][2])
        #Sétima coluna
        matrizKS [i][6] = abs(matrizKS[i][4] - matrizKS [i][3])
        #Valores máximos das colunas
        max_cols = matrizKS.max(axis=0)   
    
    #Valor encontrado pelo teste Kolmogorov-Smirnov    
    ks_calc = max(max_cols[5], max_cols[6])
    
    return matrizKS, ks_calc


def matrizQuiQuadrado(numClasses):
    grauLib2P = numClasses - 2 - 1
    grauLib3P = numClasses - 3 - 1
    matrizQuiQuadrado = np.zeros((37,2))

    #Graus de Liberdade
    for i in range(0,37,1):
        if i<30:
            matrizQuiQuadrado[i][0] = i + 1
        else:
            matrizQuiQuadrado[i][0] = matrizQuiQuadrado[i-1][0] + 10

    #Valores para cada grau de liberdade
    matrizQuiQuadrado[0][1] = 3.841
    matrizQuiQuadrado[1][1] = 5.991
    matrizQuiQuadrado[2][1] = 7.815
    matrizQuiQuadrado[3][1] = 9.488
    matrizQuiQuadrado[4][1] = 11.070
    matrizQuiQuadrado[5][1] = 12.592
    matrizQuiQuadrado[6][1] = 14.067
    matrizQuiQuadrado[7][1] = 15.507
    matrizQuiQuadrado[8][1] = 16.919
    matrizQuiQuadrado[9][1] = 18.307
    matrizQuiQuadrado[10][1] = 19.675
    matrizQuiQuadrado[11][1] = 21.026
    matrizQuiQuadrado[12][1] = 22.362
    matrizQuiQuadrado[13][1] = 23.685
    matrizQuiQuadrado[14][1] = 24.996
    matrizQuiQuadrado[15][1] = 26.296
    matrizQuiQuadrado[16][1] = 27.587
    matrizQuiQuadrado[17][1] = 28.869
    matrizQuiQuadrado[18][1] = 30.144
    matrizQuiQuadrado[19][1] = 31.410
    matrizQuiQuadrado[20][1] = 32.671
    matrizQuiQuadrado[21][1] = 33.924
    matrizQuiQuadrado[22][1] = 35.172
    matrizQuiQuadrado[23][1] = 36.415
    matrizQuiQuadrado[24][1] = 37.652
    matrizQuiQuadrado[25][1] = 38.885
    matrizQuiQuadrado[26][1] = 40.113
    matrizQuiQuadrado[27][1] = 41.337
    matrizQuiQuadrado[28][1] = 42.557
    matrizQuiQuadrado[29][1] = 43.773
    matrizQuiQuadrado[30][1] = 55.758
    matrizQuiQuadrado[31][1] = 67.505
    matrizQuiQuadrado[32][1] = 79.082
    matrizQuiQuadrado[33][1] = 90.531
    matrizQuiQuadrado[34][1] = 101.879
    matrizQuiQuadrado[35][1] = 113.145
    matrizQuiQuadrado[36][1] = 124.342

    #Verificação da estatística de teste
    if grauLib2P <= 30: 
        estatTest2P = matrizQuiQuadrado [grauLib2P - 1][1]
    elif grauLib2P <= 40:
        estatTest2P = matrizQuiQuadrado [30][1]
    elif grauLib2P <= 50: 
        estatTest2P = matrizQuiQuadrado [31][1]
    elif grauLib2P <= 60: 
        estatTest2P = matrizQuiQuadrado [32][1]
    elif grauLib2P <= 70:
        estatTest2P = matrizQuiQuadrado [33][1]
    elif grauLib2P <= 80: 
        estatTest2P = matrizQuiQuadrado [34][1]
    elif grauLib2P <= 90: 
        estatTest2P = matrizQuiQuadrado [35][1]
    else: 
        estatTest2P = matrizQuiQuadrado [36][1]   

    if grauLib3P <= 30: 
        estatTest3P = matrizQuiQuadrado [grauLib3P - 1][1]
    elif grauLib3P <= 40:
        estatTest3P = matrizQuiQuadrado [30][1]
    elif grauLib3P <= 50:
        estatTest3P = matrizQuiQuadrado [31][1]
    elif grauLib3P <= 60: 
        estatTest3P = matrizQuiQuadrado [32][1]
    elif grauLib3P <= 70: 
        estatTest3P = matrizQuiQuadrado [33][1]
    elif grauLib3P <= 80: 
        estatTest3P = matrizQuiQuadrado [34][1]
    elif grauLib3P <= 90: 
        estatTest3P = matrizQuiQuadrado [35][1]
    else:
        estatTest3P = matrizQuiQuadrado [36][1]

    return SimpleNamespace(valor2P = estatTest2P, valor3P = estatTest3P, gl2P = grauLib2P, gl3P = grauLib3P)

def resultadoFinal(maximas, isozonaEscolhida, numDist):
    #Parâmetros das distribuições
    muy, sigmay, A0, alphaII, betaII, alphaIII, betaIII, gama = parametrosDist(maximas)
    
    #Valores para isozonas
    isozonas = [['A',36.2,35.8,35.6,35.5,35.4,35.3,35,34.7,33.6,32.5,7,6.3],
            ['B',38.1,37.8,37.5,37.4,37.3,37.2,36.9,36.6,35.4,34.3,8.4,7.5],
            ['C',40.1,39.7,39.5,39.3,39.2,39.1,38.8,38.4,37.2,36,9.8,8.8],
            ['D',42,41.6,41.4,41.2,41.1,41,40.7,40.3,39,37.8,11.2,10],
            ['E',44,43.6,43.3,43.2,43,42.9,42.6,42.2,40.9,39.6,12.6,11.2],
            ['F',46,45.5,45.3,45.1,44.9,44.8,44.5,44.1,42.7,41.3,13.9,12.4],
            ['G',47.9,47.4,47.2,47,46.8,46.7,46.4,45.9,44.5,43.1,15.4,13.7],
            ['H',49.9,49.4,49.1,48.9,48.6,48.6,48.3,47.8,46.3,44.8,16.7,14.9]]
    
    #Seleção da distribuição
    if numDist == 1:
        dist = stats.gamma(a = alphaII,scale=betaII)
    if numDist == 2:
        dist = stats.gamma(a = alphaIII,loc=gama,scale=betaIII)
    if numDist == 3:
        dist = stats.lognorm(s=sigmay, scale=math.exp(muy))
    if numDist == 4:
        dist = stats.lognorm(s=sigmay, loc=A0, scale=math.exp(muy))
    if numDist == 5:
        dist = inverseGumbel(maximas)
        
    #Tempo de Retorno 
    tr5 = isozonas[isozonaEscolhida][1]
    tr10 = isozonas[isozonaEscolhida][2]
    tr15 = isozonas[isozonaEscolhida][3]
    tr20 = isozonas[isozonaEscolhida][4]
    tr25 = isozonas[isozonaEscolhida][5]
    tr30 = isozonas[isozonaEscolhida][6]
    tr50 = isozonas[isozonaEscolhida][7]
    tr100 = isozonas[isozonaEscolhida][8]
    """
    tr1000 = isozonas[isozonaEscolhida][9]
    tr10000 = isozonas[isozonaEscolhida][10]
    """
    tr6min5a50 = isozonas[isozonaEscolhida][11]
    tr6min100 = isozonas[isozonaEscolhida][12]
    
    #Precipitação para 1 dia
    if numDist<5:
        pTr5 = dist.isf(1/5)
        pTr10 = dist.isf(1/10)
        pTr15 = dist.isf(1/15)
        pTr20 = dist.isf(1/20)
        pTr25 = dist.isf(1/25)
        pTr30 = dist.isf(1/30)
        pTr50 = dist.isf(1/50)
        pTr100 = dist.isf(1/100)
        """
        pTr1000 = dist.isf(1/1000)
        pTr10000 = dist.isf(1/10000)
        """
    else:
        pTr5 = dist(5)
        pTr10 = dist(10)
        pTr15 = dist(15)
        pTr20 = dist(20)
        pTr25 = dist(25)
        pTr30 = dist(130)
        pTr50 = dist(50)
        pTr100 = dist(100)
        """
        pTr1000 = dist(1000)
        pTr10000 = dist(10000)
        """
    
    #Precipitação para 24 horas
    p24Tr5 = pTr5 * 1.096
    p24Tr10 = pTr10 * 1.096
    p24Tr15 = pTr15 * 1.096
    p24Tr20 = pTr20 * 1.096
    p24Tr25 = pTr25 * 1.096
    p24Tr30 = pTr30 * 1.096
    p24Tr50 = pTr50 * 1.096
    p24Tr100 = pTr100 * 1.096
    """
    p24Tr1000 = pTr1000 * 1.096
    p24Tr10000 = pTr10000 * 1.096
    """
    
    #Obtensão da Matriz de Chuvas Intensas
    matrizPChMax = np.zeros((9,16))
    matrizPIntMax = np.zeros((9,13))
    
    #linha com as durações das precipitações para vários tempos de retorno
    matrizPChMax[0][4] = 6
    matrizPChMax[0][5] = 12
    matrizPChMax[0][6] = 18
    matrizPChMax[0][7] = 24
    matrizPChMax[0][8] = 30
    matrizPChMax[0][9] = 36
    matrizPChMax[0][10] = 48
    matrizPChMax[0][11] = 60
    matrizPChMax[0][12] = 90
    matrizPChMax[0][13] = 120
    matrizPChMax[0][14] = 180
    matrizPChMax[0][15] = 240
    
    matrizPIntMax[0][1] = 6
    matrizPIntMax[0][2] = 12
    matrizPIntMax[0][3] = 18
    matrizPIntMax[0][4] = 24
    matrizPIntMax[0][5] = 30
    matrizPIntMax[0][6] = 36
    matrizPIntMax[0][7] = 48
    matrizPIntMax[0][8] = 60
    matrizPIntMax[0][9] = 90
    matrizPIntMax[0][10] = 120
    matrizPIntMax[0][11] = 180
    matrizPIntMax[0][12] = 240
    
    #Primeira coluna com os tempos de retorno
    matrizPChMax[1][0] = 5
    matrizPChMax[2][0] = 10
    matrizPChMax[3][0] = 15
    matrizPChMax[4][0] = 20
    matrizPChMax[5][0] = 25
    matrizPChMax[6][0] = 30
    matrizPChMax[7][0] = 50
    matrizPChMax[8][0] = 100
    
    matrizPIntMax[1][0] = 5
    matrizPIntMax[2][0] = 10
    matrizPIntMax[3][0] = 15
    matrizPIntMax[4][0] = 20
    matrizPIntMax[5][0] = 25
    matrizPIntMax[6][0] = 30
    matrizPIntMax[7][0] = 50
    matrizPIntMax[8][0] = 100
    
    #Segunda coluna com as precipitações máximas de 24h para os tempos de retorno 
    matrizPChMax[1][1] = p24Tr5
    matrizPChMax[2][1] = p24Tr10
    matrizPChMax[3][1] = p24Tr15
    matrizPChMax[4][1] = p24Tr20
    matrizPChMax[5][1] = p24Tr25
    matrizPChMax[6][1] = p24Tr30
    matrizPChMax[7][1] = p24Tr50
    matrizPChMax[8][1] = p24Tr100
    
    #Terceira coluna com os coeficientes de desagregação das Isozonas para chuvas menores que 1h em relação aos tempos de retorno
    matrizPChMax[1][2] = tr6min5a50
    matrizPChMax[2][2] = tr6min5a50
    matrizPChMax[3][2] = tr6min5a50
    matrizPChMax[4][2] = tr6min5a50
    matrizPChMax[5][2] = tr6min5a50
    matrizPChMax[6][2] = tr6min5a50
    matrizPChMax[7][2] = tr6min5a50
    matrizPChMax[8][2] = tr6min100
    
    #Quarta coluna com os coeficientes de desagregação das Isozonas para chuvas maiores que uma hora em relação aos tempos de retorno
    matrizPChMax[1][3] = tr5
    matrizPChMax[2][3] = tr10
    matrizPChMax[3][3] = tr15
    matrizPChMax[4][3] = tr20
    matrizPChMax[5][3] = tr25
    matrizPChMax[6][3] = tr30
    matrizPChMax[7][3] = tr50
    matrizPChMax[8][3] = tr100
    
    #Quinta Coluna - Precipitação com duração de 6 minutos para os tempos de retorno 
    for i in range(1,9):
        matrizPChMax[i][4] = (matrizPChMax[i][1] * matrizPChMax[i][2]) / 100
        matrizPIntMax[i][1] = matrizPChMax[i][4] / 6
        
    #Décima Segunda Coluna - Precipitação com duração de 60 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][11] = (matrizPChMax[i][1]*matrizPChMax[i][3]) / 100
        matrizPIntMax[i][8] = matrizPChMax[i][11] / 60
        
    #Sexta Coluna - Precipitaçãocom duração de 12 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][5] = (matrizPChMax[i][4]) + ((matrizPChMax[i][11] - matrizPChMax[i][4]) / (math.log(1) - math.log(0.1))) * (math.log(0.2) - math.log(0.1))
        matrizPIntMax[i][2] = matrizPChMax[i][5] / 12
    
    #Sétima Coluna - Pricipitação com duração de 18 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][6] = (matrizPChMax[i][4]) + ((matrizPChMax[i][11] - matrizPChMax[i][4]) / (math.log(1) - math.log(0.1))) * (math.log(0.3) - math.log(0.1))
        matrizPIntMax[i][3] = matrizPChMax[i][6] / 18 
    
    #Oitava Coluna - Precipitação com duração de 24 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][7] = (matrizPChMax[i][4]) + ((matrizPChMax[i][11] - matrizPChMax[i][4]) / (math.log(1) - math.log(0.1))) * (math.log(0.4) - math.log(0.1))
        matrizPIntMax[i][4] = matrizPChMax[i][7] / 24 
            
    #Nona Coluna - Precipitação com duração de 30 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][8] = (matrizPChMax[i][4]) + ((matrizPChMax[i][11] - matrizPChMax[i][4]) / (math.log(1) - math.log(0.1))) * (math.log(0.5) - math.log(0.1))
        matrizPIntMax[i][5] = matrizPChMax[i][8] / 30 
    
    
    #Décima Coluna - Precipitação com duração de 36 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][9] = (matrizPChMax[i][4]) + ((matrizPChMax[i][11] - matrizPChMax[i][4]) / (math.log(1) - math.log(0.1))) * (math.log(0.6) - math.log(0.1))
        matrizPIntMax[i][6] = matrizPChMax[i][9] / 36
        
        
    #Décima Primeira Coluna - Precipitação com duração de 48 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][10] = (matrizPChMax[i][4]) + ((matrizPChMax[i][11] - matrizPChMax[i][4]) / (math.log(1) - math.log(0.1))) * (math.log(0.8) - math.log(0.1))
        matrizPIntMax[i][7] = matrizPChMax[i][10] / 48 
        
        
    #Décima Terceira Coluna - Precipitação com duração de 90 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][12] = (matrizPChMax[i][11]) + ((matrizPChMax[i][1] - matrizPChMax[i][11]) / (math.log(24) - math.log(1))) * (math.log(1.5) - math.log(1))
        matrizPIntMax[i][9] = matrizPChMax[i][12] / 90
        
    #Décima Quarta Coluna - Precipitação com duração de 120 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][13] = (matrizPChMax[i][11]) + ((matrizPChMax[i][1] - matrizPChMax[i][11]) / (math.log(24) - math.log(1))) * (math.log(2) - math.log(1))
        matrizPIntMax[i][10] = matrizPChMax[i][13] / 120    
        
    #Décima Quinta Coluna - Precipitação com duração de 180 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][14] = (matrizPChMax[i][11]) + ((matrizPChMax[i][1] - matrizPChMax[i][11]) / (math.log(24) - math.log(1))) * (math.log(3) - math.log(1))
        matrizPIntMax[i][11] = matrizPChMax[i][14] / 180    
        
    #Décima sexta Coluna - Precipitação com duração de 240 minutos para os tempos de retorno
    for i in range(1,9):
        matrizPChMax[i][15] = (matrizPChMax[i][11]) + ((matrizPChMax[i][1] - matrizPChMax[i][11]) / (math.log(24) - math.log(1))) * (math.log(4) - math.log(1))
        matrizPIntMax[i][12] = matrizPChMax[i][15] / 240    
        
    #Cálculo do Parâmetro C da Função Gamma II
    pObservado = len(maximas)/5
    
    TRWilken = float(pObservado)
    
    if numDist<5:
        pTrWilken = dist.isf(1/TRWilken)
    else:
        pTrWilken = dist(TRWilken)
        
    p24TrWilken = pTrWilken*1.096

    TrW = (((tr100 - tr5) * (TRWilken - 5)) / 95) + tr5
    p1hWilken = (TrW*p24TrWilken)/100
    p6TrWilken = (p24TrWilken * tr6min5a50) / 100
    p240TrWilken = (p1hWilken) + ((p24TrWilken - p1hWilken) / (math.log(24) - math.log(1))) * (math.log(4) - math.log(1))
    
    I6minWilken = p6TrWilken/6
    I240minWilken = p240TrWilken/240
    I1 = I6minWilken
    I2 = I240minWilken
    I3 = math.pow((I1*I2),0.5)
    
    I1hWilken = p1hWilken/60
    
    if (I3 > I1hWilken):
        durPI3 = 59
        pI3 = (p6TrWilken) + ((p1hWilken - p6TrWilken) / (math.log(1) - math.log(0.1))) * (math.log(durPI3 / 60) - math.log(0.1))
        intPI3 = pI3 / durPI3
        while (I3 > intPI3):
            durPI3 = durPI3 - 0.01
            pI3 = (p6TrWilken) + ((p1hWilken - p6TrWilken) / (math.log(1) - math.log(0.1))) * (math.log(durPI3 / 60) - math.log(0.1))
            intPI3 = pI3 / durPI3
    
    if (I3 < I1hWilken):
        durPI3 = 61
        pI3 = (p1hWilken) + ((p24TrWilken - p1hWilken) / (math.log(24) - math.log(1))) * (math.log(durPI3 / 60) - math.log(1))
        intPI3 = pI3 / durPI3
        while (I3 < intPI3):
            durPI3 = durPI3 + 0.01
            pI3 = (p1hWilken) + ((p24TrWilken - p1hWilken) / (math.log(24) - math.log(1))) * (math.log(durPI3 / 60) - math.log(1))
            intPI3 = pI3 / durPI3
            
    parametroC = (math.pow(durPI3,2)-1440)/(246-(2*durPI3))
    
    
    
    #Cálculo para Determinação dos Parâmetros A, B e N
    matrizABN = np.zeros((96,10))
    
    #Primeira Coluna da MatriABN
    for i in range(0,12):
        matrizABN[i][0] = 5
    
    for i in range(12,24):
        matrizABN[i][0] = 10
        
    for i in range(24,36):
        matrizABN[i][0] = 15
        
    for i in range(36,48):
        matrizABN[i][0] = 20
        
    for i in range(48,60):
        matrizABN[i][0] = 25
        
    for i in range(60,72):
        matrizABN[i][0] = 30
        
    for i in range(72,84):
        matrizABN[i][0] = 50
        
    for i in range(84,96):
        matrizABN[i][0] = 100
    
    #Segunda Coluna da MatriABN
    for i in range(0,96,12):
        matrizABN[i][1] = 6
    
    for i in range(1,96,12):
        matrizABN[i][1] = 12
        
    for i in range(2,96,12):
        matrizABN[i][1] = 18
        
    for i in range(3,96,12):
        matrizABN[i][1] = 24
        
    for i in range(4,96,12):
        matrizABN[i][1] = 30
        
    for i in range(5,96,12):
        matrizABN[i][1] = 36
        
    for i in range(6,96,12):
        matrizABN[i][1] = 48
        
    for i in range(7,96,12):
        matrizABN[i][1] = 60
        
    for i in range(8,96,12):
        matrizABN[i][1] = 90
        
    for i in range(9,96,12):
        matrizABN[i][1] = 120
        
    for i in range(10,96,12):
        matrizABN[i][1] = 180
        
    for i in range(11,96,12):
        matrizABN[i][1] = 240
        
    #Terceira Coluna da MatriABN
    for i in range(0,96):
        if i < 12:
            matrizABN[i][2] = math.log(matrizPIntMax[1][i+1])
        if 12 <= i < 24:
            matrizABN[i][2] = math.log(matrizPIntMax[2][i-11])
        if 24 <= i < 36:
            matrizABN[i][2] = math.log(matrizPIntMax[3][i-23])
        if 36 <= i < 48:
            matrizABN[i][2] = math.log(matrizPIntMax[4][i-35])
        if 48 <= i < 60:
            matrizABN[i][2] = math.log(matrizPIntMax[5][i-47])
        if 60 <= i < 72:
            matrizABN[i][2] = math.log(matrizPIntMax[6][i-59])
        if 72 <= i < 84:
            matrizABN[i][2] = math.log(matrizPIntMax[7][i-71])
        if i >= 84:
            matrizABN[i][2] = math.log(matrizPIntMax[8][i-83])
        
    #Quarta Coluna da MatriABN
    for i in range(0,96):
        if i < 12:
            matrizABN[i][3] = math.log(5 - 2)
        if 12 <= i < 24:
            matrizABN[i][3] = math.log(10 - 2)
        if 24 <= i < 36:
            matrizABN[i][3] = math.log(15 - 2)
        if 36 <= i < 48:
            matrizABN[i][3] = math.log(20 - 2)
        if 48 <= i < 60:
            matrizABN[i][3] = math.log(25 - 2)
        if 60 <= i < 72:
            matrizABN[i][3] = math.log(30 - 2)
        if 72 <= i < 84:
            matrizABN[i][3] = math.log(50 - 2)
        if i >= 84:
            matrizABN[i][3] = math.log(100 - 2)
    
    #Quinta Coluna da MatriABN
    for i in range(0,96):
        matrizABN[i][4] = math.log(matrizABN[i][1] + parametroC)
    
    #Sexta Coluna da MatriABN
    for i in range(0,96):
        matrizABN[i][5] = math.pow(matrizABN[i][3], 2)
    
    #Sétima Coluna da MatriABN
    for i in range(0,96):
        matrizABN[i][6] = math.pow(matrizABN[i][4],2)
    
    #Oitava Coluna da MatriABN
    for i in range(0,96):
        matrizABN[i][7] = matrizABN[i][2] * matrizABN[i][3]
    
    #Nona Coluna da MatriABN
    for i in range(0,96):
        matrizABN[i][8] = matrizABN[i][2] * matrizABN[i][4]
    
    #Décima Coluna da MatriABN
    for i in range(0,96):
        matrizABN[i][9] = matrizABN[i][3] * matrizABN[i][4]
    
    #Somatório das Colunas 3 a 10
    somatórioColunas = matrizABN.sum(axis=0)
    Lni = somatórioColunas[2]
    LnTS = somatórioColunas[3]
    LntC = somatórioColunas[4]
    BQuadrado = somatórioColunas[5]
    CQuadrado = somatórioColunas[6]
    AvezesB = somatórioColunas[7]
    AvezesC = somatórioColunas[8]
    BvezesC = somatórioColunas[9]
    
    #Cálculo dos Determinantes da Matriz
    DetP = ((96 * BQuadrado * CQuadrado) + (LnTS * BvezesC * LntC) + (LntC * LnTS * BvezesC) - (LnTS * LnTS * CQuadrado) - (96 * BvezesC * BvezesC) - (LntC * BQuadrado * LntC))
    
    DetAzero = ((Lni * BQuadrado * CQuadrado) + (LnTS * BvezesC * AvezesC) + (LntC * AvezesB * BvezesC) - (LnTS * AvezesB * CQuadrado)
                        - (Lni * BvezesC * BvezesC) - (LntC * BQuadrado * AvezesC));
    
    DetAHum = ((96 * AvezesB * CQuadrado) + (Lni * BvezesC * LntC) + (LntC * LnTS * AvezesC) - (Lni * LnTS * CQuadrado) - (96 * BvezesC * AvezesC)
                        - (LntC * AvezesB * LntC));
    
    DetAdois = ((96 * BQuadrado * AvezesC) + (LnTS * AvezesB * LntC) + (Lni * LnTS * BvezesC) - (LnTS * LnTS * AvezesC)
                        - (96 * AvezesB * BvezesC) - (Lni * BQuadrado * LntC));
    
    #Determinação dos Parâmetros
    parametroA = math.exp(DetAzero / DetP)
    
    parametroB = DetAHum / DetP
    
    parametroN = -DetAdois / DetP
        
    #Parâmtro S
    #Método do Mínimo Qui-Quadrado
    parametroSTeste = -10
    minSomaQui = 99999
    parametroSFinal = 0
    somaQuiParS = 1000
    minQuiQuad = np.zeros((96,5))
    
    for i in range(0,96):
        minQuiQuad[i][0] = matrizABN[i][0]
        minQuiQuad[i][1] = matrizABN[i][1]
    
    #Inserção dos valores de Taborga observados
    for i in range(0,96):
        if i < 12:
            minQuiQuad[i][2] = matrizPIntMax[1][i+1]
        if 12 <= i < 24:
            minQuiQuad[i][2] = matrizPIntMax[2][i-11]
        if 24 <= i < 36:
            minQuiQuad[i][2] = matrizPIntMax[3][i-23]
        if 36 <= i < 48:
            minQuiQuad[i][2] = matrizPIntMax[4][i-35]
        if 48 <= i < 60:
            minQuiQuad[i][2] = matrizPIntMax[5][i-47]
        if 60 <= i < 72:
            minQuiQuad[i][2] = matrizPIntMax[6][i-59]
        if 72 <= i < 84:
            minQuiQuad[i][2] = matrizPIntMax[7][i-71]
        if i >= 84:
            minQuiQuad[i][2] = matrizPIntMax[8][i-83]
    
    #Início dos testes
    while somaQuiParS <= minSomaQui or somaQuiParS==float('Inf') or math.isnan(somaQuiParS):
        for i in range(0,96):
            minQuiQuad[i][3] = (parametroA * ((minQuiQuad[i][0] + parametroSTeste)**parametroB)) / (math.pow((minQuiQuad[i][1] + parametroC), parametroN))
        
        for i in range(0,96):
            minQuiQuad[i][4] = (math.pow((minQuiQuad[i][2] - minQuiQuad[i][3]), 2)) / minQuiQuad[i][3]
            
        somaColunasMin = minQuiQuad.sum(axis=0)
        somaQuiParS = somaColunasMin[4]
        
        if somaQuiParS < minSomaQui:
            minSomaQui = somaQuiParS
            parametroSFinal = parametroSTeste
            
        parametroSTeste = parametroSTeste + 0.005
   
    
    return matrizPChMax, matrizPIntMax, parametroA, parametroB, parametroC, parametroN, parametroSFinal


