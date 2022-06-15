import pandas as pd
import math
import numpy as np
import csv
from math import pi
from numpy import fft
import time #tiempo de espera

###############################################################################################

##se inicia respuesta impulsional en la matriz A
##Cargamos S11
S = []
#F = np.zeros((201,256)) # matriz final de freq
Pos ='B'
for n_tr in range(16):
    for n_rx in range(16):
        ##definiendo archivo de lectura
        na_tr = n_tr + 1
        na_rx = n_rx + 1
        name_file =  'results'+Pos+ '_p'+ str(na_tr) + '_' + str(na_rx)+ '.csv'
        
        ##Leyendo datos de csv
        v=pd.read_csv(name_file,header = 0)
        vm=v.iloc[:,0].values 
        vp=v.iloc[:,1].values
        f=v.iloc[:,2].values

        Sp=10** (vm/20) * np.exp(1j * vp * pi/180)
        
        ##Convirtiendo datos a str
        for i in range(len(Sp)):
            if Sp[i].imag>=0:
                S.append(str(Sp[i].real)+'+'+str(Sp[i].imag)+'j')
            else:
                S.append(str(Sp[i].real)+str(Sp[i].imag)+'j')

#guardando archivos S y Freq
S=np.array(S).reshape(256,501).T
S1=pd.DataFrame(data=S)
S1.to_csv('Fantoma_1cm_2dBm_H2_Pos_10'+Pos+'.csv',index=False,header=False,)
#np.savetxt('results_Freq_'+Pos+'.csv', f, delimiter=",")

