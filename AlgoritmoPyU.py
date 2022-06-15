import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as ma
from math import pi
from scipy.interpolate import interp1d
import winsound


def agru_senal(n1,n2,na):
    M1 = [] #columnas(señales en vacio) agrupadas
    M2 = [] #columnas(señales con tumor) agrupadas
    Pos = 'A'
    for i in range(1,na+1):
        for j in range(1,na+1):
            name_file = n1 +'/' + 'results'+Pos+ '_p'+ str(i) + '_' + str(j)+ '.csv' ##definiendo archivo de lectura
            name_file2 = n2 +'/' + 'results'+Pos+ '_p'+ str(i) + '_' + str(j)+ '.csv'
            
            v=pd.read_csv(name_file,header = 0)##Leyendo csv de file vacio
            vm=v.iloc[:,0].values #mag
            vp=v.iloc[:,1].values #fase
            f=v.iloc[:,2].values #freq
            Sp=10** (vm/20) * np.exp(1j * vp * pi/180) # conversion fase a complejo
    
            v2=pd.read_csv(name_file2,header = 0)##Leyendo csv de file con tumor
            vm2=v2.iloc[:,0].values #mag
            vp2=v2.iloc[:,1].values #fase
            Sp2=10** (vm2/20) * np.exp(1j * vp2 * pi/180) # conversion fase a complejo
            
            for m in range(len(Sp)):   ## Agrupando señales y convirtiendo a str       
                if Sp[m].imag>=0: #M1
                    M1.append(str(Sp[m].real)+'+'+str(Sp[m].imag)+'j')
                else:
                    M1.append(str(Sp[m].real)+str(Sp[m].imag)+'j')
                if Sp2[m].imag>=0: #M2
                    M2.append(str(Sp2[m].real)+'+'+str(Sp2[m].imag)+'j')
                else:
                    M2.append(str(Sp2[m].real)+str(Sp2[m].imag)+'j')
 
    M1=np.array(M1).reshape(na*na,len(vm)).T #matriz (señales en vacio) agrupadas
    M1=pd.DataFrame(data=M1)
    M2=np.array(M2).reshape(na*na,len(vm)).T #matriz (señales con tumor) agrupadas
    M2=pd.DataFrame(data=M2)
    f1=pd.DataFrame(data=f).to_numpy()*10**6
    rows=np.shape(M2)[0]
    cols=np.shape(M2)[1]
    
    #M1.to_csv(n1+'_agrupado.csv',index=False,header=False,) #guardando archivos S y Freq
    #M2.to_csv(n2+'_agrupado.csv',index=False,header=False,)
    #f1.to_csv('Frequency_range_'+str(len(vm))+'.csv',index=False, header=False)
    
    M1=M1.to_numpy().astype(complex) 
    M2=M2.to_numpy().astype(complex) 
    return M1,M2,f1,rows,cols
    
    
def env(lenv, M, rows, cols):  # envolvente
    Xw = M.real
    Env = np.zeros((rows,cols),dtype = 'complex')
    for i in range(cols):
        #FFT
        Xf = np.fft.fft(Xw[:,i], 1024)
        Xf = Xf[0:int(1024/2+1)]
        #IFFT
        x=Xf[np.arange(0,lenv)]
        c=np.zeros((1,len(Xf)-lenv))
        sx=np.concatenate((x,c),axis=None)
        fp=np.flipud(sx)
        Xfp=np.concatenate((sx,fp),axis=None)
        Xh = np.fft.ifft(Xfp)
        Xh = Xh[0:int(1024/2+1)]
        Env[:,i]=2*Xh[0:rows]
    return Env

def herm(f,Env_w, Env_wo):  #señal calibrada
    # hermitiano
    f_res = (max(f)-min(f))/len(f)
    Nzeros = np.round((min(f)/f_res)-1)
    fs = 2*max(f) 
    N = np.size(Env_w,axis=1)
    Wo = np.concatenate((np.transpose(np.zeros((N,int(Nzeros)),dtype=int)), Env_wo, np.flipud(Env_wo), np.transpose(np.zeros((N,int(Nzeros-1)),dtype=int))),axis=0)
    W  = np.concatenate((np.transpose(np.zeros((N,int(Nzeros)),dtype=int)), Env_w, np.flipud(Env_w), np.transpose(np.zeros((N,int(Nzeros-1)),dtype=int))),axis=0)
    IFFT_Without1 = np.fft.ifft(Wo,axis=0)
    IFFT_With1 = np.fft.ifft(W, axis=0)
    Time = range(len(IFFT_Without1))/fs
    rows2 = len(IFFT_With1)
    
    #hamming
    Window = np.array(np.hamming(len(IFFT_With1)))
    IFFT_With = np.zeros((rows2,cols),dtype = 'complex')
    IFFT_Without = np.zeros((rows2,cols),dtype = 'complex')
    for i in range(cols):
        IFFT_Without[:,i] = IFFT_Without1[:,i]*Window
        IFFT_With[:,i] = IFFT_With1[:,i]*Window
    
    #calibrado
    Calibrated_With = IFFT_With - IFFT_Without
    return Calibrated_With, Time, rows2

def removal_clutter(Calibrated_With, cols, rows2, na):  #removiendo clutter
    Average = np.zeros((rows2,na),dtype='complex')
    l = 1
    for k in range(1,cols+1):
        Average[:,l-1] = Average[:,l-1]+Calibrated_With[:,k-1]
        if k%na == 0:
            l = l+1
    Average = Average/na
    
    New_With = np.zeros((rows2,cols),dtype='complex')
    l = 1
    for k in range(1,cols+1):
        New_With[:,k-1] = Calibrated_With[:,k-1]-Average[:,l-1]
        if k%na == 0:
            l = l+1
    Removed_Signal = abs(New_With)
    return Removed_Signal

def grid_antena(Pos,minx,maxx,miny,maxy,MAXx,MAXy,cols,H): #distancia de antenas
    b = Pos[:,0].T
    a = Pos[:,1].T
    c = Pos[:,2].T
    #[b1,a1,c1] = cart2sph(b,a,c)
    AntennaLocations_x = b
    AntennaLocations_y = a
    AntennaLocations_z = c
    X,Y,Z = np.meshgrid(np.linspace(minx,maxx,MAXx),np.linspace(miny,maxy,MAXy),np.linspace(-2,9,MAXy)) #grilla
    X = X.flatten()
    Y = Y.T.flatten()
    Z = Z.T.flatten()
    Distance = np.zeros((len(X),cols))
    for k in range(cols):
        for m in range(len(X)):
            Distance[m,k] = ma.sqrt((X[m]-AntennaLocations_x[k])**2+(Y[m]-AntennaLocations_y[k])**2+(H-AntennaLocations_z[k])**2)
    return Distance, X, Y


def das(Distance,cols ,Er_in,Er_max,ner):  #interpolacion, promedio de permitividades y das
    Intensity_p =np.zeros(len(Distance))
    for Er in range(Er_in,Er_max+1,ner):
        vdelt = (cv/ma.sqrt(Er))
        TmR = np.zeros((len(Distance),cols))
        for k in range(cols):
            for l in range(len(Distance)):
                TmR[l,k] = 2*Distance[l,k]/vdelt
        #interpolacion
        InterpolatedData = []
        for m in range(cols):
            f = interp1d(Time,Removed_Signal[:,m])
            InterpolatedData.append(f(TmR[:,m]))
        InterpolatedData=np.array(InterpolatedData).T
        #das
        IntensityValues=np.zeros(len(InterpolatedData))
        for n in range(len(InterpolatedData)):
            IntensityValues[n]=sum(InterpolatedData[n,:])
        Intensity=(IntensityValues/max(IntensityValues))**2
        Intensity_p = Intensity_p + Intensity
    Intensity_prom = Intensity_p/ner
    return Intensity_prom

def das_cf(Distance,cols ,Er_in,Er_max,ner):  #interpolacion, promedio de permitividades y das-cf
    Intensity_p =np.zeros(len(Distance))
    for Er in range(Er_in,Er_max+1,ner):
        vdelt = (cv/ma.sqrt(Er))
        TmR = np.zeros((len(Distance),cols))
        for k in range(cols):
            for l in range(len(Distance)):
                TmR[l,k] = 2*Distance[l,k]/vdelt
        #interpolacion
        InterpolatedData = []
        for m in range(cols):
            f = interp1d(Time,Removed_Signal[:,m])
            InterpolatedData.append(f(TmR[:,m]))
        InterpolatedData=np.array(InterpolatedData).T
        ##DAS-cf
        #signal 2
        InterpolatedData2 = np.zeros((len(InterpolatedData),cols))
        for o in range(cols-1):
            InterpolatedData2[:,o]=InterpolatedData[:,o]**2
        #sumatorias de señales
        IntensityValues=np.zeros(len(InterpolatedData))
        IntensityValues2=np.zeros(len(InterpolatedData2))
        for n in range(len(InterpolatedData)):
            IntensityValues[n]=sum(InterpolatedData[n,:])
            IntensityValues2[n]=sum(InterpolatedData2[n,:])
        Intensity1=(IntensityValues/max(IntensityValues))
        Intensity2=(IntensityValues2/max(IntensityValues2))
        #eq final
        Intensity=(Intensity2**3)/(Intensity1)
        
        Intensity_p = Intensity_p + Intensity
    Intensity_prom = Intensity_p/ner
    return Intensity_prom

def dmas(Distance,cols ,Er_in,Er_max,ner):  #interpolacion, promedio de permitividades y dmas
    Intensity_p =np.zeros(len(Distance))
    for Er in range(Er_in,Er_max+1,ner):
        vdelt = (cv/ma.sqrt(Er))
        TmR = np.zeros((len(Distance),cols))
        for k in range(cols):
            for l in range(len(Distance)):
                TmR[l,k] = 2*Distance[l,k]/vdelt
        #interpolacion
        InterpolatedData = []
        for m in range(cols):
            f = interp1d(Time,Removed_Signal[:,m])
            InterpolatedData.append(f(TmR[:,m]))
        InterpolatedData=np.array(InterpolatedData).T
        ##DMAS
        #signal 2
        InterpolatedData2 = np.zeros((len(InterpolatedData),cols))
        for o in range(cols-1):
            InterpolatedData2[:,o]=np.sign(InterpolatedData[:,o])*(abs(InterpolatedData[:,o])**(1/2))
        #sumatorias de señales
        IntensityValues=np.zeros(len(InterpolatedData))
        IntensityValues2=np.zeros(len(InterpolatedData2))
        for n in range(len(InterpolatedData)):
            IntensityValues[n]=sum(InterpolatedData[n,:])
            IntensityValues2[n]=sum(InterpolatedData2[n,:])
        Intensity1=(IntensityValues/max(IntensityValues))
        Intensity2=(IntensityValues2/max(IntensityValues2))
        #eq final
        Intensity=(Intensity1**2-Intensity2)/2
        
        Intensity_p = Intensity_p + Intensity
    Intensity_prom = Intensity_p/ner
    return Intensity_prom

# =============================================================================
# =============================================================================

for HH in range(9,10):
    ## tabla de configuracion
    n1='datos vacio -10 dBm' #nombre de carpeta de señales en vacio
    n2='pos_4_H_3_pot_-10dBm' #nombre de carpeta de señales con tumor
    lenv=40 # muestras para generar enventanado
    H=HH # altura a detectar (cm)
    Er_in=6 ; Er_max=6; ner=1  # rangos de permitividad
    afi=das_cf #algoritmo a usar (das,dmas,das_cf)
    
    cv=2.99e10 #velocidad de luz (cm/s)
    na=16 #antenas usadas
    Pos1=  np.array([[7.25,0.7,1],[6.8,0.7,2.7],[4.9,0.7,5.3],[2.8,0.7,6],[-1,7.25,1.2],[-1,5.4,3.1],[-1,3.4,5],[-1,1.6,6.1],[-7.25,-1.8,1.25],[-6.2,-1.8,3.4],[-4.9,-1.8,5.1],[-3,-1.8,6],[2,-7.3,1.4],[1.8,-6.8,3.5],[2,-5.8,5],[2.1,-2.8,6]])
    Pos=np.tile(Pos1,(na,1)) #posicion antenas
    MAXx = 40 #pixeles eje x (30)
    MAXy = 30 #pixeles eje y (20)
    minx=-9; maxx=9 #rango eje x imagen (cm)
    miny=-9; maxy=9 #rango eje y imagen (cm)
    
    #generando funciones
    [M1,M2,f,rows,cols] = agru_senal(n1,n2,na) #agrupamiento de señales de VNA
    Env_wo = env(lenv,M1,rows,cols) #envolvente without
    Env_w = env(lenv,M2,rows,cols) #envolvente with
    [Calibrated_With, Time, rows2] = herm(f,Env_w, Env_wo) #señal calibrada
    Removed_Signal = removal_clutter(Calibrated_With, cols, rows2, na) #removiendo clutter
    [Distance,X,Y] = grid_antena(Pos,minx,maxx,miny,maxy,MAXx,MAXy,cols,H)  #posicion de antenas
    Intensity_prom = afi(Distance,cols, Er_in,Er_max,ner) #interpolacion, promedio de permitividad y algoritmo de formacion imagen
    
    #graficando imagen de tumor
    plt.scatter(X,Y,70,Intensity_prom) 
    plt.title( n2+' prueba [Altura='+str(H)+'cm; Er='+str(Er_in)+ ' ('+afi.__name__+')]')
    plt.xlabel('Desplazamiento (cm)')
    plt.ylabel('Desplazamiento (cm)')
    plt.show()

#zumbador
winsound.Beep(frequency =1000, duration = 500)


