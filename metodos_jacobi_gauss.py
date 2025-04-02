import numpy as np
import pandas as pd

def sanssenfeld(A):
    """
    Função para verificar a condição de convergência de um sistema utilizando o critério de Sassenfeld.
    A função retorna True se o sistema convergir e False caso contrário.
    """
    n = np.shape(A)[0]  # Obtém o número de linhas (ou colunas) da matriz A
    B = np.zeros(n)  # Inicializa o vetor B com zeros

    # Inicializa o primeiro valor de B
    B[0] = np.sum(np.abs(A[0])) - np.abs(A[0][0])

    # Calcula os valores do vetor B para cada linha
    for i in range(1, n):
        a = 0
        for j in range(0, n):
            # Verifica se i != j e se a < j
            if i != j and a < j:
                B[i] += abs(A[i][j]) * B[a]  # Acumula o produto dos coeficientes
            elif i != j:
                B[i] += abs(A[i][j])  # Acumula os valores absolutos dos coeficientes

    # Pega o maior valor de B
    max_x = max(B)
    
    return max_x < 1  # Retorna True se a soma for menor que 1 (convergência), senão retorna False


def converge(A):
    """
    Função que verifica se a matriz A é diagonalmente dominante.
    Retorna False se a matriz não for diagonalmente dominante, caso contrário, retorna True.
    """
    n = np.shape(A)[0]  # Obtém o número de linhas (ou colunas) da matriz A

    for i in range(0, n):
        sum = 0
        # Soma os elementos fora da diagonal
        for j in range(0, n):
            if i != j:
                sum += A[i][j]
        # Verifica se o valor da diagonal é maior que a soma dos valores fora dela
        if abs(A[i][i]) < sum:
            return False  # Se a matriz não for diagonalmente dominante, retorna False

    return True  # Se for diagonalmente dominante, retorna True


def jacobi(A, b, x0, tol, N):
    """
    Função que resolve um sistema linear Ax = b usando o método de Jacobi.
    A função retorna a solução x, e os pontos de iteração gerados durante o processo.
    A função também verifica a convergência usando os métodos de Sassenfeld e diagonalmente dominante.
    """

    n = np.shape(A)[0]  # Obtém o número de incógnitas (tamanho do vetor de solução)
    x = np.zeros(n)  # Inicializa o vetor de solução x com zeros
    it = 0  # Inicializa o contador de iterações
    pontos_iteracao = []  # Lista para armazenar os pontos gerados a cada iteração

    # Inicia as iterações do método de Jacobi
    while it < N:
        it += 1  # Incrementa o número de iterações

        # Realiza a iteração de Jacobi
        for i in range(0, n):
            x[i] = b[i]  # Começa com o valor de b[i]
            for j in range(0, n):
                if i != j:
                    x[i] -= A[i, j] * x0[j]  # Subtrai o produto dos coeficientes
            x[i] /= A[i, i]  # Divide pela diagonal de A para isolar x[i]

        # Armazena o ponto gerado nesta iteração
        
        pontos_iteracao.append(np.copy(x))  # Adiciona a solução da iteração à lista

        # Verifica a tolerância: se a diferença entre x e x0 for menor que tol, convergiu
        if np.linalg.norm(x - x0, np.inf) < tol:
            return x, pontos_iteracao  # Retorna a solução e os pontos de iteração

        # Prepara para a próxima iteração, atualizando x0 com a solução atual
        x0 = np.copy(x)

    return x, pontos_iteracao  # Retorna a solução após o número máximo de iterações

def gauss_seidel(A, b, x0, tol, N):
    n = len(A)
    x = np.copy(x0)
    pontos_iteracao = []
    
    for it in range(N):
        x_old = np.copy(x)
        
        for i in range(n):
            sum_ = b[i]
            for j in range(n):
                if i != j:
                    sum_ -= A[i, j] * x[j]
            x[i] = sum_ / A[i, i]
        
        pontos_iteracao.append(np.copy(x))
        
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, pontos_iteracao
    
    return x, pontos_iteracao

def calcular_erros(A, b, x0, tol, N):
    x_real = np.linalg.solve(A, b)  # Solução exata do sistema
    
    x_jacobi, pontos_jacobi = jacobi(A, b, x0, tol, N)
    x_gauss, pontos_gauss = gauss_seidel(A, b, x0, tol, N)
    
    tabela = []
    for i in range(min(len(pontos_jacobi), len(pontos_gauss))):
        erro_jacobi = np.linalg.norm(pontos_jacobi[i] - x_real, np.inf)
        erro_gauss = np.linalg.norm(pontos_gauss[i] - x_real, np.inf)
        tabela.append([i+1, erro_jacobi, erro_gauss])
    
    df = pd.DataFrame(tabela, columns=["Iteração", "Erro Jacobi", "Erro Gauss-Seidel"])
    return df

