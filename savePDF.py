# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:12:57 2020

@author: engjo
"""

from fpdf import FPDF
import pandas as pd
import math

"""
                UFCA - IDF
        Laboratório de Recursos Hídricos (LAHI)
        Universidade Federal do Cariri (UFCA)
        
1 - Daddos do Posto Pluviométrico

Estado: CE          Cidade: Abaiara
Código: 739046      Nome: Abaiara
Quantidade de anos da amostra: 40
Latitude: -7.3666666        Longitude: -39.05


    Tabela 1 - Dados do Posto Pluviométrico
    Ano     | Precipitação  | Ano   | Precipitação
    
    
2 - Estatística Descritiva

Média: 88.53            Desvio Padrão: 24.44
Variância: 597.12       Coeficiente de Assimetria: 0.469

3 - Equação IDF

Isozona: A          Distribuição de Probabilidade: Gumbel

3.1 - Parametros da equação IDF

A: 19.15
B: 0.123
C: 15.945
S: -2.08
N: 0.76


        Tabela 2 - Intensidades das Chuvas



"""

class Save_PDF:
    def __init__(self, estado, cidade,codigo, posto, lat, long, qntAnos, precipitacao, md, var, desvPad, coefAssimetria,dist, isozona, pA, pB, pC, pS, pN, arquivo):
        self.estado = estado
        self.cidade = cidade
        self.codigo = codigo
        self.posto = posto
        self.lat = lat
        self.long = long
        self.qntAnos = qntAnos
        self.precipitacao = precipitacao
        self.md = md
        self.var = var
        self.desvPad = desvPad
        self.coefAssimetria = coefAssimetria
        self.dist = dist
        self.isozona = isozona
        self.pA = pA
        self.pB = pB
        self.pC = pC
        self.pS = pS
        self.pN = pN
        self.arquivo = arquivo
        self.titulo = 'UFCA - IDF'
        self.subtitulo1 = 'Universidade Federal do Cariri - UFCA'
        self.subtitulo2 = 'Laboratório de HidroInformática - LAHI'
        self.subtitulo3 = 'hidroinfo.ufca.edu.br'
        self.texto1 = 'Dados do Posto Pluviométrico'
        self.texto2 = 'Estatística Descritiva'
        self.texto3 = 'Distribuição de Probabilidade de Ajuste e Isozona'
        self.texto4 = 'Informações da equação IDF'
        self.texto5 = 'Parametros da equação IDF'
        self.texto6 = "Equação IDF"
        self.tabela1 = 'Dados de chuva do Posto Pluviométrico'
        self.tabela2 = 'Intensidades das Chuvas'
        self.data = pd.DataFrame(data=self.precipitacao, columns=['Ano', 'Precipitacao'])
      
    def txtTitulo(self, texto):
        # Times 20
        self.pdf.set_font('Times', 'B', 20)
        # Titulo
        self.pdf.cell(0, 10, txt = texto, ln=1, align='C')
    
    def subtitulo(self, texto):
        # Times 20
        self.pdf.set_font('Times', 'B', 14)
        # Titulo
        self.pdf.cell(0, 6, txt = texto, ln=1, align='C')
    
    def titulo_capitulo(self, num, texto):
        # Times 12
        self.pdf.set_font('Times', 'B', 12)
        # Titulo
        self.pdf.cell(0, 6, txt ='{} - {}'.format(num, texto), ln=1, align='L')
        # Salto de línea
        self.pdf.ln(2)
    
    def titulo_tabela(self, num, texto):
        # Times 12
        self.pdf.set_font('Times', '', 12)
        # Titulo
        self.pdf.cell(0, 6, txt ='Tabela {} - {}'.format(num, texto), ln=1, align='L')
        # Salto de línea
        self.pdf.ln(2)
        
    def txt(self, alinha, texto, parametro,line):
        # Times 12
        self.pdf.set_font('Times', '', 12)
        # Titulo
        self.pdf.cell(alinha, 6, '%s : %s' % (texto, parametro), ln=line, align='L')
    

    def main(self):
        self.pdf = FPDF()
        self.pdf.add_page()
        """
        self.pdf.image('icon.png', x = 170, y =10, w = 20, h = 20, type = 'PNG')
        self.pdf.image('LAHI.jpg', x = 10, y =10, w = 20, h = 20, type = 'JPG')
        """
        self.txtTitulo(self.titulo)
        self.subtitulo(self.subtitulo1)
        self.subtitulo(self.subtitulo2)
        self.subtitulo(self.subtitulo3)
        self.pdf.ln(6)
        self.titulo_capitulo(1, self.texto1)
        self.pdf.ln(2)
        self.txt(75, 'Estado', self.estado,0)
        self.txt(75, 'Cidade', self.cidade,1)
        self.txt(75,'Código', self.codigo,0)
        self.txt(75,'Nome do Posto', self.posto,1)
        self.txt(75,'Latitude', self.lat,0)
        self.txt(75,'Longitude', self.long,1)
        self.txt(75,'Quantidade de Anos', self.qntAnos,1)
        self.pdf.ln(6)
        self.pdf.cell(40)
        self.titulo_tabela(1, self.tabela1)
        
        self.pdf.set_font('Times', '', 12)
        self.pdf.set_fill_color( r = 200 )
        self.pdf.cell(40)
        self.pdf.cell(15, 10, 'Ano', border ='TB', ln=0, align='C', fill=True)
        self.pdf.cell(35, 10, 'Precipitação (mm)', border ='TB', ln=0, align='C',fill=True)
        self.pdf.cell(15, 10, 'Ano', border ='TB',ln= 0, align='C',fill=True)
        self.pdf.cell(35, 10, 'Precipitação (mm)', border ='TB', ln=2, align='C',fill=True)
        self.pdf.cell(-65)
        self.pdf.set_font('Times', '', 12)
        self.pdf.set_fill_color ( r = 0)
        qntLinhas = math.ceil(len(self.data)/2)
        iteracao = qntLinhas -1
        for i in range(0, iteracao):
            self.pdf.cell(15, 10, '%s' % (str(self.data.loc[i, 'Ano'])), border =0, ln=0, align='C')
            self.pdf.cell(35, 10, '%s' % (str(self.data.loc[i, 'Precipitacao'])),border = 0,ln= 0, align='C')
            self.pdf.cell(15, 10, '%s' % (str(self.data.loc[i+qntLinhas, 'Ano'])), border =0, ln=0, align='C')
            self.pdf.cell(35, 10, '%s' % (str(self.data.loc[i+qntLinhas, 'Precipitacao'])), border =0, ln=2, align='C')
            self.pdf.cell(-65)
        try:
            self.pdf.cell(15, 10, '%s' % (str(self.data.loc[iteracao, 'Ano'])), border ='B', ln=0, align='C')
            self.pdf.cell(35, 10, '%s' % (str(self.data.loc[iteracao, 'Precipitacao'])),border = 'B',ln= 0, align='C')
            self.pdf.cell(15, 10, '%s' % (str(self.data.loc[qntLinhas+iteracao, 'Ano'])), border ='B', ln=0, align='C')
            self.pdf.cell(35, 10, '%s' % (str(self.data.loc[qntLinhas+iteracao, 'Precipitacao'])), border ='B', ln=2, align='C')
            self.pdf.cell(-65)
        
        except:
            self.pdf.cell(15, 10, '', border ='B', ln=0, align='C')
            self.pdf.cell(35, 10, '', border ='B', ln=2, align='C')
            self.pdf.cell(-65)
    
       
        self.pdf.ln(10)
        self.titulo_capitulo(2, self.texto2)
        self.pdf.ln(2)
        self.txt(75,'Média', self.md,0)
        self.txt(75,'Desvio Padrão', self.desvPad,1)
        self.txt(75,'Variância', self.var,0)
        self.txt(75,'Coeficiente de Assimetria', self.coefAssimetria,1)
        self.pdf.ln(6)
        self.titulo_capitulo(3, self.texto3)
        self.pdf.ln(2)
        self.txt(75,'Distribuição de Probabilidade', self.dist,2)
        self.txt(75,'Isozona', self.isozona,1)
        self.pdf.ln(6)
        self.titulo_capitulo(4, self.texto4)
        self.pdf.ln(2)
        self.titulo_capitulo(4.1, self.texto5)
        self.pdf.ln(2)
        self.txt(30, 'A', self.pA,0)
        self.txt(30,'B', self.pB,0)
        self.txt(30,'C', self.pC,0)
        self.txt(30,'S', self.pS,0)
        self.txt(30,'N', self.pN,2)
        self.pdf.ln(6)
        """
        self.titulo_capitulo(4.2, self.texto6)
        self.pdf.ln(2)
        self.pdf.image('i.png', x = 75, w = 50, h = 25, type = 'PNG')
        self.pdf.cell(0, 6, 'Sendo:', ln=2, align='L')
        self.pdf.cell(10)
        self.pdf.cell(0, 6, 'Tr: Tempo de retorno em anos', ln=2, align='L')
        self.pdf.cell(0, 6, 't: Tempo de duração da chuva em min', ln=2, align='L')
        self.pdf.cell(0, 6, 'i: Intensidade da chuva em mm/min', ln=2, align='L')
        """
        

    
        self.pdf.output(self.arquivo)

