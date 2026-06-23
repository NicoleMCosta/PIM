import numpy as np

def dir_grad(gy, gx):
    return np.arctan2(gy,gx)

def max_locais(magnitude, direcao):
    maximos = np.zeros(magnitude.shape, dtype=np.float32)
    
    angulo = np.rad2deg(direcao)
    linhas, colunas = magnitude.shape
    
    for i in range(1, linhas - 1):
        for j in range(1, colunas - 1):
            vizinho_1 = 0
            vizinho_2 = 0
            
            # diagonal superior direita e inferior esquerda
            if (22.5 < angulo[i,j]<= 67.5) or (-157.5<= angulo[i,j] < -122.5):
                vizinho_1 = magnitude[i-1,j+1]
                vizinho_2 = magnitude[i+1,j-1]

            # vertical 
            elif (67.5 < angulo[i,j]<= 112.5) or (-112.5 <= angulo[i,j] < -67.5):
                vizinho_1 = magnitude[i-1,j]
                vizinho_2 = magnitude[i+1,j]

            # diagonal superior esquerda e inferior direita
            elif (112.5 < angulo[i,j] <= 157.5) or (-67.5 <= angulo[i,j] < -22.5):
                vizinho_1 = magnitude[i-1,j-1]
                vizinho_2 = magnitude[i+1,j+1]

            # horizontal
            else:
                vizinho_1 = magnitude[i,j-1]
                vizinho_2 = magnitude[i,j+1]

            if (magnitude[i, j] >= vizinho_1) and (magnitude[i, j] >= vizinho_2):
                maximos[i, j] = magnitude[i, j]
            else:
                maximos[i, j] = 0

    return maximos

def prewitt_scharr(image,kernel_x, kernel_y):
    m, n = image.shape
    gx = np.zeros((m,n), dtype=np.float32)
    gy = np.zeros((m,n), dtype=np.float32)
    magnitude = np.zeros((m,n), dtype=np.float32)
    
    for linha in range(1, m-1): 
        for coluna in range(1, n-1):
            vizinhanca = image[linha-1:linha+2, coluna-1:coluna+2].astype(np.float32)
            
            gx[linha, coluna]= np.sum(vizinhanca * kernel_x)
            gy[linha, coluna] = np.sum(vizinhanca * kernel_y)
            
            magnitude[linha, coluna] = np.sqrt(gx[linha, coluna]**2 + gy[linha, coluna]**2)
    
    dir = dir_grad(gy,gx)
    maximos= max_locais(magnitude, dir)    
    return maximos
