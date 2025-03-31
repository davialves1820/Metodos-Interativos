import numpy as np
from jacobi import jacobi  # Importa a função 'jacobi' de um módulo externo
import matplotlib.pyplot as plt  # Importa a biblioteca para plotar gráficos

def main():
    # Define o sistema de equações: Ax = b
    # A matriz A representa os coeficientes das variáveis no sistema linear
    A = np.array([[2, 1], [3, 4]], dtype=float)  # Matriz dos coeficientes
    b = np.array([1, -1], dtype=float)  # Vetor dos termos independentes
    x0 = np.array([0, 0], dtype=float)  # Chute inicial para as soluções
    tol = 0.0001  # Tolerância para a convergência do método de Jacobi
    N = 20  # Número máximo de iterações permitidas para o método de Jacobi

    try:
        # Chama a função jacobi para resolver o sistema linear Ax = b
        # A função retorna a solução aproximada e os pontos das iterações
        solucao, pontos_jacobi = jacobi(A, b, x0, tol, N)
        print("Solução encontrada:", solucao)  # Exibe a solução final

    except ValueError as e:
        # Se ocorrer um erro durante a execução do método de Jacobi, ele será capturado aqui
        print("Erro:", e)
        return  # Sai da função main caso ocorra um erro

    # Verifica se `pontos_jacobi` é uma lista válida com pelo menos um ponto
    if isinstance(pontos_jacobi, list) and len(pontos_jacobi) > 0:
        pontos_jacobi = np.array(pontos_jacobi)  # Converte para um array numpy
    else:
        # Se a lista de pontos estiver vazia ou não for válida, imprime uma mensagem e define como None
        print("Nenhum ponto gerado pelo método de Jacobi.")
        pontos_jacobi = None  # Define como None para evitar erro no gráfico

    # Definir o intervalo de valores de x1 para o gráfico
    x = np.linspace(0, 2, 400)  # Gera um intervalo de 400 pontos entre 0 e 2

    # Define as equações das retas a serem plotadas no gráfico
    f1 = (1 - 2*x)  # Equação 1: 2x1 + x2 = 1 -> x2 = (1 - 2x1)
    f2 = (-1 - 3*x) / 4  # Equação 2: 3x1 + 4x2 = -1 -> x2 = (-1 - 3x1) / 4

    # Cria o gráfico com tamanho de figura personalizado
    plt.figure(figsize=(8, 6))

    # Plota as duas equações no gráfico
    plt.plot(x, f1, label='2x1 + x2 = 1', color='blue')  # Linha azul para a primeira equação
    plt.plot(x, f2, label='3x1 + 4x2 = -1', color='red')  # Linha vermelha para a segunda equação

    # Se pontos de iteração foram gerados, plota-os no gráfico
    if pontos_jacobi is not None and pontos_jacobi.shape[1] == 2:  
        # Plota os pontos das iterações de Jacobi como pontos verdes
        plt.scatter(pontos_jacobi[:, 0], pontos_jacobi[:, 1], color='green', marker='o', s=50, label='Iterações de Jacobi')

    # Adiciona legenda, rótulos e título ao gráfico
    plt.xlabel('x1')  # Rótulo para o eixo x
    plt.ylabel('x2')  # Rótulo para o eixo y
    plt.title('Convergência do Método de Jacobi')  # Título do gráfico
    plt.legend()  # Exibe a legenda
    plt.grid()  # Exibe a grade no gráfico

    # Exibe o gráfico gerado
    plt.show()

# Executa o programa principal quando o script é rodado
if __name__ == "__main__":
    main()  
