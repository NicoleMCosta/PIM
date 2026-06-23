import numpy as np


def dir_grad(gy, gx):
    return np.arctan2(gy,gx)

def max_locais(magnitude, direcao):
    maximos = np.zeros_like(magnitude)

    angulo = np.rad2deg(direcao)
    angulo[angulo < 0] += 180

    linhas, colunas = magnitude.shape


    for i in range(1, linhas-1):
        for j in range(1, colunas-1):
            ang = angulo[i,j]
            atual = magnitude[i,j]

            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                n1 = magnitude[i,j-1]
                n2 = magnitude[i,j+1]
            elif 22.5 <= ang < 67.5:
                n1 = magnitude[i-1,j+1]
                n2 = magnitude[i+1,j-1]
            elif 67.5 <= ang < 112.5:
                n1 = magnitude[i-1,j]
                n2 = magnitude[i+1,j]
            else:
                n1 = magnitude[i-1,j-1]
                n2 = magnitude[i+1,j+1]

            if atual >= n1 and atual >= n2:
                maximos[i,j] = atual

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
