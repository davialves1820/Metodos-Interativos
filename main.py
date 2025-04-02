import numpy as np
from metodos_jacobi_gauss import calcular_erros

def main():
    # Define o sistema de equações: Ax = b
    # A matriz A representa os coeficientes das variáveis no sistema linear
    A = np.array([[80, 0, 30, 10], [0, 80, 10, 10], [16, 20, 60, 72], [4, 0, 0, 8]], dtype=float)  # Matriz dos coeficientes
    b = np.array([40, 27, 31, 2], dtype=float)  # Vetor dos termos independentes
    x0 = np.array([0.5, 0.2, 0.2, 0], dtype=float)  # Chute inicial para as soluções
    tol = 0.000001  # Tolerância para a convergência do método de Jacobi
    N = 100   # Número máximo de iterações permitidas para o método de Jacobi

    A1 = np.array([
    [0.7071, 0.0000, 0.0000, -0.8500, 0.0000, 0.0000, 0.1000, 0.0500, 0.0300],
    [0.7071, 1.0500, 0.0000, 0.4500, 1.0000, 0.1000, 0.1000, 0.0500, 0.0200],
    [0.0000, 0.0000, 1.0000, 0.0000, -1.0500, 0.1000, 0.2000, 0.1000, 0.0500],
    [0.0000, 0.0000, 0.0000, -0.9000, 0.4500, 1.0500, 0.3000, 0.2000, 0.1000],
    [0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 0.1500, 1.0500, 0.2500, 0.1500],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0500, 0.2500, -1.1000, 0.0500],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.1000, 0.4500, 0.1500],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4500, 1.1000, -1.2000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.1000]], dtype=float)

    # Vetor B
    B1 = np.array([0, -1000, 0, 0, 500, 0, 0, -500, 0], dtype=float)
    m0 =np.zeros(len(B1), dtype=float)

    try:
        # Chama a função jacobi para resolver o sistema linear Ax = b
        # A função retorna a solução aproximada e os pontos das iterações
        #solucao, pontos_jacobi = jacobi(A, b, x0, tol, N)
        tabela_erros = calcular_erros(A1, B1, m0, tol, N)
        tabela_erros = tabela_erros.round(6) 
        print(tabela_erros.to_string(index=False))

    except ValueError as e:
        # Se ocorrer um erro durante a execução do método de Jacobi, ele será capturado aqui
        print("Erro:", e)
        return  # Sai da função main caso ocorra um erro


# Executa o programa principal quando o script é rodado
if __name__ == "__main__":
    main()  
