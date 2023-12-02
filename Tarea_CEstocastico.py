#!/usr/bin/env python
# coding: utf-8

# # Calculo Estocastico

# ## Tarea Simulaciones

# #### Aldo Mendoza Rebollar 201930811

# Cargamos las posibles librerias de uso
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy import optimize as opt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from __future__ import division
from math import *
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from datetime import datetime, timedelta
from IPython.display import display, Math


# ## Ejercicio 1

# ## $ dX(t)=2w(t)+1dt $

# ## Solucion $ X(t)=w^2(t)+k $

# ## Aplicamos el cambio en lo ya hecho

# In[2]:


# Función de derivación estocástica
def diffusion_process(dt, X):
    dW = np.random.normal(0, np.sqrt(dt))  # Incremento estocástico.
    dX = 2 * dW + 1 * dt  # Derivada de la ecuación estocástica.
    return X + dX

# Función de la solución exacta
def exact_solution(t, X0):
    return t + X0 - 2 * np.cumsum(np.random.normal(0, np.sqrt(t[1] - t[0]), len(t)))

# Función de Euler-Maruyama para la difusión
def euler_maruyama_diffusion(t0, T, N, X0):
    t = np.linspace(t0, T, N + 1)
    X = X0
    resultados = [X]

    for i in range(N):
        dt = t[i + 1] - t[i]
        dX = diffusion_process(dt, X)
        X = dX
        resultados.append(X)

    return t, resultados


# En este código, se está modelando un proceso estocástico utilizando el método de Euler-Maruyama para resolver una ecuación diferencial estocástica específica. Aquí hay una explicación matemática y numérica de lo que se está haciendo:
# 
# 1. **Ecuación Estocástica:** La ecuación estocástica que se está resolviendo es \(dX(t) = 2W(t) + 1dt\), donde \(W(t)\) es un proceso de Wiener (o movimiento browniano) y \(dt\) es el incremento de tiempo.
# 
# 2. **Método de Euler-Maruyama:** El método de Euler-Maruyama es una técnica numérica utilizada para simular soluciones de ecuaciones diferenciales estocásticas. En este caso, la función `euler_maruyama_diffusion` implementa el método de Euler-Maruyama para simular la evolución del proceso estocástico a lo largo del tiempo.
# 
# 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Definir los parámetros
t0 = 0
T = 1
N = 1000
X0 = 100
k = 3


# Realizar simulaciones
n_simulations =5
plt.figure(figsize=(10, 6))

for i in range(n_simulations):
    t, resultados = euler_maruyama_diffusion(t0, T, N, X0)
    plt.plot(t, resultados, label=f'Simulación {i + 1}')

plt.xlabel('Tiempo')
plt.ylabel('Resultado')
plt.title('Simulaciones de Euler-Maruyama para la Difusión')
plt.legend()
plt.show()


# 3. **Función de Derivación Estocástica:** La función `diffusion_process` calcula el incremento estocástico en la variable \(X\) en un intervalo de tiempo \(\Delta t\). En este caso, se utiliza el método de Euler para aproximar la solución.
# 
# 4. **Solución Exacta:** La función `exact_solution` proporciona una solución exacta para la ecuación estocástica. Utiliza la acumulación de sumas de incrementos de Wiener para construir la solución exacta.
# 
# 5. **Simulaciones:** Se realizan simulaciones numéricas utilizando el método de Euler-Maruyama para varias trayectorias del proceso estocástico. Estas simulaciones se comparan con la solución exacta y se visualizan en un gráfico.
# 
# El problema modelado aquí es un proceso estocástico donde la variable \(X(t)\) evoluciona estocásticamente en el tiempo debido a la presencia de un término estocástico (\(2W(t)\)) en la ecuación diferencial. El objetivo es simular y comparar estas trayectorias estocásticas con la solución exacta.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
t0 = 0
T = 1
N = 1000
X0 = 0  # Condición inicial específica
k = 3

# Realizar simulaciones
n_simulations = 7
plt.figure(figsize=(10, 6))

# Plot de la solución exacta
t_exact = np.linspace(t0, T, N + 1)
exact_sol = exact_solution(t_exact, X0)
plt.plot(t_exact, exact_sol, label='Solución Exacta', linestyle='--', color='black')

# Plot de simulaciones de Euler-Maruyama
for i in range(n_simulations):
    t, resultados = euler_maruyama_diffusion(t0, T, N, X0)
    plt.plot(t, resultados, label=f'Simulación {i + 1}')

plt.xlabel('Tiempo')
plt.ylabel('Resultado')
plt.title('Solución Exacta vs. Simulaciones de Euler-Maruyama')
plt.legend()
plt.show()


# ## Ejercicio 2

# ### Generar M trayectorias del proceso Ornstein-Uhlenbeck y graficarlos junto con $xExp(- \alpha * t)$ para $ 0 \leq t \leq T $

# ### $dX(t)= (\alpha -\beta*X(t))dt+ \sigma *dW(t)$ 

# $X_t= (\alpha/ \beta)*(1-exp(-\beta*(t-s)))+exp(-\beta*(t-s)*X_s + exp(-\beta*(t))* \int_{s}^{t} e^{\beta*t} \, dWt$

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.1
beta = 0.5
sigma = 0.2
X0 = 2
T = 5
dt = 0.01
pasos = int(T / dt)

# Function for Euler-Maruyama simulation
def ornstein_uhlenbeck_euler_maruyama(alpha, beta, sigma, X0, T, dt, rng):
    X = np.zeros(pasos + 1)
    X[0] = X0

    for i in range(1, pasos + 1):
        dW = np.sqrt(dt) * rng.normal(0, 1)
        dX = (alpha - beta * X[i-1]) * dt + sigma * dW
        X[i] = X[i-1] + dX

    return np.arange(0, T, dt), X[:-1]

# Simulate and plot multiple times
N_simulations = 5

for i in range(N_simulations):
    rng = np.random.RandomState(seed=123 + i)
    t, resultados = ornstein_uhlenbeck_euler_maruyama(alpha, beta, sigma, X0, T, dt, rng)
    plt.plot(t, resultados, label=f'Simulación {i + 1}')

# Plot analytical solution using the last simulation results
analytical_solution = X0 + (alpha/beta - X0) * (1 - np.exp(-beta * t))
plt.plot(t, analytical_solution, label=r'$X_0 + (\frac{\alpha}{\beta} - X_0) (1 - e^{-\beta t})$', color='orange', linestyle='dashed')

plt.xlabel('Tiempo')
plt.ylabel('X(t)')
plt.title('Simulación de Ornstein-Uhlenbeck usando Euler-Maruyama')
plt.legend()
plt.show()


# El proceso estocástico conocido como "Ecuación Diferencial Estocástica de Ornstein-Uhlenbeck". 
# Se define como:
# 
# $dX(t) = (\alpha - \beta X(t)) dt + \sigma dW(t) $
# 
# Donde:
# - $X(t)$ es el proceso estocástico que estamos modelando.
# - $\alpha, \beta,  $ y $\sigma$  son parámetros constantes.
# - $dW(t)$ es el incremento de un proceso de Wiener (ruido blanco) que representa la componente estocástica del proceso.
# 
# **Solución Numérica (Método de Euler-Maruyama):**
# Para resolver numéricamente esta ecuación, se utiliza el método de Euler-Maruyama. Este método es una extensión del método de Euler para ecuaciones diferenciales ordinarias y se adapta a procesos estocásticos. La idea es discretizar el tiempo y aproximar la solución en pasos sucesivos, teniendo en cuenta la componente estocástica.
# 
# **Solución Analítica:**
# Para la solución analítica, se busca una expresión cerrada para \(X(t)\). En el caso del proceso de Ornstein-Uhlenbeck, la solución analítica toma la forma:
# 
# $X(t) = X_s + \left(\frac{\alpha}{\beta} - X_s\right) \left(1 - e^{-\beta t}\right) + \sigma \int_0^t e^{-\beta (t-s)} dW(s) $
# 
# Donde:
# - $X_0$ es la condición inicial del proceso.
# 
# La parte $ \sigma \int_0^t e^{-\beta (t-s)} dW(s)$ representa la componente estocástica y está relacionada con el proceso de Wiener.
# 
# Ambas soluciones, numérica y analítica, proporcionan formas de entender y simular el comportamiento del proceso estocástico en el tiempo, teniendo en cuenta tanto la dinámica determinística
# $\alpha - \beta X(t)  $ como la componente estocástica $\sigma \ dW(t)$.

# # Ejercicio 3

# ## Simulación de Puente Browniano

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(n, Delta):
    trayMB = np.zeros(n + 1)
    trayMB[0] = 0

    for i in range(1, n + 1):
        trayMB[i] = trayMB[i - 1] + np.random.normal(0, 1) * Delta

    return trayMB


def brownian_bridge(x, y, n, t0, T):
    Delta = (T - t0) / n
    t0 = t0 / n

    bridge = np.zeros(n + 1)

    W = brownian_motion(n, Delta)

    for i in range(n + 1):
        bridge[i] = x + W[i] - ((i - 1) / n) * (W[n] - y + x)

    times = np.arange(0, T + Delta, Delta)
    plt.plot(times, bridge)
    plt.ylabel("Puente Browniano")
    plt.xlabel("Tiempo")
    plt.show()

    return bridge


# Parámetros
x = 0
y = 0
n = 1000  # Número de pasos de tiempo
t0 = 0
T = 1.0  # Tiempo total de la simulación

# Llamada a la función
puente_browniano = brownian_bridge(x, y, n, t0, T)


# El problema matemático que estamos abordando aquí está relacionado con la construcción de un "puente Browniano". Un puente Browniano es una trayectoria estocástica que conecta dos puntos fijos, y en este caso, estamos construyendo un puente Browniano entre los puntos $(t_0, x)$ y $(T, y)$.
# 
# La construcción se realiza utilizando el proceso de Wiener, también conocido como movimiento Browniano. La función `brownian_motion` genera un movimiento Browniano discretizado con $n$ pasos de tiempo y un incremento de tiempo $\Delta$.
# 
# La función `brownian_bridge` utiliza el movimiento Browniano generado para construir el puente Browniano. La idea es ajustar la trayectoria Browniana para que empiece en $(t_0, x)$ y termine en 
# $(T, y)$.
# 
# **Solución Numérica:**
# El método numérico utilizado aquí es simular el movimiento Browniano y luego ajustar la trayectoria para que sea un puente Browniano entre dos puntos fijos.
# 
# Este tipo de construcción de puentes Brownianos se utiliza en finanzas y probabilidad para modelar caminos aleatorios que cumplen ciertas restricciones.
# 
# En resumen, la solución numérica involucra la simulación de un movimiento Browniano y la posterior construcción de un puente Browniano entre dos puntos específicos.

# ## Ejercicio 4

# ## Modelación por Euler-Maruyama

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

def diffusion_process(t, X, mu, sigma):
    # Definición de la ecuación de difusión.
    dW = np.random.normal(0, np.sqrt(t[1] - t[0]))  # Incremento estocástico.
    dX = mu * X * (t[1] - t[0]) + sigma * X * dW
    return X + dX

def euler_maruyama_diffusion(t0, T, N, X0, mu, sigma, n_simulations):
    # Parámetros:
    # t0: Tiempo inicial.
    # T: Tiempo final.
    # N: Número de pasos de discretización.
    # X0: Valor inicial del proceso de difusión.
    # mu: Tasa de crecimiento determinística.
    # sigma: Coeficiente de difusión.
    # n_simulations: Número de simulaciones de Monte Carlo.

    # Construye el vector de tiempos.
    t = np.linspace(t0, T, N + 1)

    # Inicializa una matriz para almacenar los resultados de todas las simulaciones.
    resultados = np.zeros((n_simulations, N + 1))

    # Realiza las simulaciones de Monte Carlo.
    for i in range(n_simulations):
        # Inicializa el proceso de difusión para cada simulación.
        X = X0

        # Itera a través de los pasos de tiempo.
        for j in range(N):
            dX = diffusion_process(t[j:j+2], X, mu, sigma)
            X = dX
            resultados[i, j + 1] = X

    return t, resultados

def plot_simulations(t, resultados, n_simulations, xlim=None, ylim=None):
    # Grafica las trayectorias
    plt.figure(figsize=(12, 8))
    for i in range(n_simulations):
        plt.plot(t, resultados[i])
    plt.xlabel('Tiempo')
    plt.ylabel('Precio de la Acción')
    plt.title('Trayectorias de Precio de la Acción en Monte Carlo')
    plt.legend()

    # Aplica zoom si los límites se especifican
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.show()

# Parámetros del modelo
t0 = 0
T = 1
N = 365
X0 = 100
mu = 0.05 
sigma = 0.04
n_simulations = 100  # Número de simulaciones

# Realiza las simulaciones
t, resultados = euler_maruyama_diffusion(t0, T, N, X0, mu, sigma, n_simulations)

# Grafica las trayectorias con zoom
plot_simulations(t, resultados, n_simulations, xlim=(0.001, 1), ylim=(80, 120))


# En este código, se modela un proceso estocástico mediante el método de Monte Carlo para simular soluciones de una ecuación diferencial estocástica (SDE). La SDE es una ecuación de difusión que sigue la forma:
# 
# $dX(t) = \mu X(t) \Delta t + \sigma X(t) dW(t) $
# 
# - $ \mu $: Tasa de crecimiento determinística.
# - $ \sigma $: Coeficiente de difusión.
# - $ X(t) $: Proceso de difusión.
# - $ \Delta t $: Incremento de tiempo.
# - $ dW(t) $: Incremento del proceso de Wiener.
# 
# El método de Euler-Maruyama se utiliza para discretizar esta ecuación, y la función `diffusion_process` implementa la ecuación de difusión estocástica:
# 
# $dX(t) = \mu X(t) \Delta t + \sigma X(t) dW(t) $
# 
# La función `euler_maruyama_diffusion` realiza múltiples simulaciones de Monte Carlo utilizando el método de Euler-Maruyama. Cada simulación proporciona trayectorias del proceso de difusión a lo largo del tiempo.
# 
# 
# En resumen, el código modela y simula el comportamiento de un proceso estocástico definido por una ecuación diferencial estocástica mediante el método de Monte Carlo y el método de Euler-Maruyama.

# ## Ejercicio 5
# 

# Calcular el precio de una opción call

# Primero sacamos la data de una acción real.

# In[8]:


import yfinance as yf
import numpy as np

def get_log_returns(symbols, start_date, end_date):
    return {symbol: np.log(1 + yf.download(symbol, start=start_date, end=end_date)['Close'].pct_change()) for symbol in symbols}

# Lista de símbolos de acciones
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Especifica las fechas de inicio y finalización
start_date = "2022-11-30"
end_date =datetime.now().strftime('%Y-%m-%d')

# Obtener los rendimientos logarítmicos
DATA = get_log_returns(symbols, start_date, end_date)

# Visualizar los primeros registros de los rendimientos logarítmicos
for symbol, log_returns in DATA.items():
    print(f"Rendimientos Logarítmicos para {symbol}:\n{log_returns.head()}\n")


# In[9]:


pd.DataFrame(DATA)


# Calculamos m y v

# In[10]:


import yfinance as yf
import numpy as np
import pandas as pd

def get_log_returns(symbols, start_date, end_date):
    return {symbol: np.log(1 + yf.download(symbol, start=start_date, end=end_date)['Close'].pct_change()) for symbol in symbols}

def calculate_summary_statistics(log_returns_dict):
    summary_data = []

    for symbol, log_returns in log_returns_dict.items():
        mean_returns = log_returns.mean()
        variance_returns = log_returns.var()

        summary_data.append({
            'Symbol': symbol,
            'Mean Returns': mean_returns,
            'Variance Returns': variance_returns
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Lista de símbolos de acciones
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Especifica las fechas de inicio y finalización
start_date = "2022-11-30"
end_date = datetime.now().strftime('%Y-%m-%d')


# Obtener los rendimientos logarítmicos
log_returns_dict = get_log_returns(symbols, start_date, end_date)

# Calcular las estadísticas resumen
summary_df = calculate_summary_statistics(log_returns_dict)

# Visualizar el DataFrame con las estadísticas resumen
print("Estadísticas Resumen:")
print(summary_df)


# In[11]:


import yfinance as yf
import numpy as np
import pandas as pd

def get_log_returns(symbols, start_date, end_date):
    return {symbol: np.log(1 + yf.download(symbol, start=start_date, end=end_date)['Close'].pct_change()) for symbol in symbols}

def calculate_summary_statistics(log_returns_dict):
    summary_data = []

    for symbol, log_returns in log_returns_dict.items():
        mean_returns = log_returns.mean()
        variance_returns = log_returns.var()
        delta_t = 1 / len(log_returns)

        sigma_returns = np.sqrt(variance_returns / delta_t)
        beta_returns = (mean_returns / delta_t) + (0.5 * sigma_returns)

        summary_data.append({
            'Symbol': symbol,
            'Beta': beta_returns,
            'Sigma': sigma_returns
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Lista de símbolos de acciones
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Especifica las fechas de inicio y finalización
start_date = "2022-11-30"
end_date = datetime.now().strftime('%Y-%m-%d')

# Obtener los rendimientos logarítmicos
log_returns_dict = get_log_returns(symbols, start_date, end_date)

# Calcular las estadísticas resumen
summary_df = calculate_summary_statistics(log_returns_dict)

# Visualizar el DataFrame con las estadísticas resumen
print("Estadísticas Resumen Modificadas:")
summary_df


# In[12]:


import yfinance as yf
import numpy as np
import pandas as pd

def get_log_returns(symbols, start_date, end_date):
    return {symbol: np.log(1 + yf.download(symbol, start=start_date, end=end_date)['Close'].pct_change()) for symbol in symbols}

def calculate_summary_statistics(log_returns_dict):
    summary_data = []

    for symbol, log_returns in log_returns_dict.items():
        mean_returns = log_returns.mean()
        variance_returns = log_returns.var()
        delta_t = 1 / len(log_returns)
        k = yf.download(symbol, start=start_date, end=end_date)['Close'].mean()  # Calcula el promedio de los valores reales

        sigma_returns = np.sqrt(variance_returns / delta_t)
        beta_returns = (mean_returns / delta_t) + (0.5 * sigma_returns)

        summary_data.append({
            'Symbol': symbol,
            'Beta': beta_returns,
            'Sigma': sigma_returns,
            'k': k
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Lista de símbolos de acciones
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Especifica las fechas de inicio y finalización
start_date = "2022-11-30"
end_date = datetime.now().strftime('%Y-%m-%d')

# Obtener los rendimientos logarítmicos
log_returns_dict = get_log_returns(symbols, start_date, end_date)

# Calcular las estadísticas resumen
summary_df = calculate_summary_statistics(log_returns_dict)

# Visualizar el DataFrame con las estadísticas resumen
print("Estadísticas Resumen Modificadas:")
summary_df


# ## Utilizamos a Black Sholes

# In[14]:


t0 = 0
T = 1
N = 365
X0 = 100
mu = summary_df['Beta'][0]
sigma = summary_df['Sigma'][0]
k=summary_df['k'][0]
n_simulations = 10  # Número de simulaciones

# Realiza las simulaciones
t, resultados = euler_maruyama_diffusion(t0, T, N, X0, mu, sigma, n_simulations)

# Grafica las trayectorias con zoom
plot_simulations(t, resultados, n_simulations)


# In[20]:


def plot_simulations(t, resultados, k, n_simulations):
    plt.figure(figsize=(12, 8))

    # Asegurar que todas las variables tengan la misma longitud
    k_values = np.full_like(t, k)

    for i in range(n_simulations):
        plt.plot(t, resultados[i][:len(t)])

    plt.plot(t, k_values, label='k', linestyle='--', color='red')  # Añadir la línea de k
    plt.xlabel('Tiempo')
    plt.ylabel('Precio de la Acción')
    plt.title('Trayectorias de Precio de la Acción en Monte Carlo')
    plt.legend()

    plt.show()


# In[21]:


t0 = 0
T = 1
N = 365
X0 = 100
mu = summary_df['Beta'][0]
sigma = summary_df['Sigma'][0]
n_simulations = 100  # Número de simulaciones

# Calcular k (promedio del precio de la acción)
k = np.mean(yf.download("AAPL", start="2022-11-30", end="2023-11-30")['Close'])

# Realiza las simulaciones y grafica
t, resultados = euler_maruyama_diffusion(t0, T, N, X0, mu, sigma, n_simulations)
plot_simulations(t, resultados, k, n_simulations)


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Función de derivación estocástica
def diffusion_process(dt, X, mu, sigma):
    dW = np.random.normal(0, np.sqrt(dt))  # Incremento estocástico.
    dX = mu * X * dt + sigma * X * dW
    return X + dX

# Función de Euler-Maruyama para la difusión
def euler_maruyama_diffusion(t0, T, N, X0, mu, sigma, n_simulations):
    t = np.linspace(t0, T, N + 1)
    resultados = []

    for _ in range(n_simulations):
        X = X0
        resultados_simulacion = [X]

        for i in range(N):
            dt = t[i + 1] - t[i]
            dX = diffusion_process(dt, X, mu, sigma)
            X = dX
            resultados_simulacion.append(X)

        resultados.append(resultados_simulacion)

    # Asegurar que todos los resultados tengan la misma longitud
    min_length = min(len(simulacion) for simulacion in resultados)
    resultados = [simulacion[:min_length] for simulacion in resultados]

    return t[:min_length], resultados

# Función para graficar simulaciones
def plot_simulations(t, resultados, k, n_simulations):
    plt.figure(figsize=(12, 8))

    # Asegurar que todas las variables tengan la misma longitud
    k_values = np.full_like(t, k)

    for i in range(n_simulations):
        plt.plot(t, resultados[i])

    plt.plot(t, k_values, label='k', linestyle='--', color='black')  # Añadir la línea de k
    plt.xlabel('Tiempo')
    plt.ylabel('Precio de la Acción')
    plt.title('Trayectorias de Precio de la Acción en Monte Carlo')
    plt.legend()

    plt.show()

# Parámetros del modelo

symbol = "AAPL"
latest_data = yf.download(symbol, period="1d")
k = latest_data['Close'].iloc[-1]
t0 = 0
T = 1
N = 365
X0 = k
mu = summary_df['Beta'][0]
sigma = summary_df['Sigma'][0]
n_simulations = 100  # Número de simulaciones

# Calcular k (promedio del precio de la acción)

start_date = "2022-12-1"
end_date = datetime.now().strftime('%Y-%m-%d')
#k = yf.download(symbol, start=start_date, end=end_date)['Close'].mean()




# Realiza las simulaciones y grafica
t, resultados = euler_maruyama_diffusion(t0, T, N, X0, mu, sigma, n_simulations)
plot_simulations(t, resultados, k, n_simulations)


# In[23]:


# Función para calcular el valor de una opción call al tiempo T
def calculate_call_option_value(X_T, k):
    return max(X_T - k, 0)

# Función para calcular el promedio de las diferencias mayores a 0
def calculate_average_positive_differences(results, k):
    differences = []

    for simulacion in results:
        X_T = simulacion[-1]
        call_option_value = calculate_call_option_value(X_T, k)
        differences.append(call_option_value)

    # Filtrar solo las diferencias mayores a 0
    positive_differences = [diff for diff in differences if diff > 0]

    # Calcular el promedio de las diferencias positivas
    average_positive_difference = np.mean(positive_differences)

    return average_positive_difference


# In[24]:


average_positive_difference = calculate_average_positive_differences(resultados, k)

display(Math(f"E\\left[(S_T - k)^+\\right]: {average_positive_difference}"))




# #### Calculemos ahora el valor del call como $C = e^{-\mu (T)} \cdot \mathbb{E}\left[(S_T - K)^+\right]$
# 

# 
# 
# Donde:
# - $ C$ es el precio de una opción de compra (call option).
# - $ \mu $ es el rendimiento esperado (drift).
# - $ T $ es el tiempo de vencimiento de la opción.
# - $ t $ es el tiempo actual.
# - $ S_T $ es el precio del activo en el tiempo de vencimiento.
# - $ K $ es el precio de ejercicio de la opción.
# 
# 

# In[25]:


summary_df['Beta'][0]


# $C = e^{-(0.3560241354338681) (1)} \cdot (95.06797245396903)$

# In[26]:


mu = summary_df['Beta'][0]
T = 1
K = average_positive_difference

# Calcular C
C = np.exp(-mu * T) * 95.06797245396903 
print(f"Precio de la opción Call = {C}")


# ## Simulaciones de tarea

# ## Simulación I(t)

# #### Proceso de Wiener

# Este fragmento de código simula el proceso de Wiener generando incrementos aleatorios en cada paso de tiempo $t_i$. La trayectoria del proceso de Wiener $W(t)$ se calcula mediante la acumulación de estos incrementos.
# 
# En términos matemáticos, esto se expresa como:
# 
# 
# $W(t_i) = W(t_{i-1}) + \sqrt{\Delta t} \cdot Z_i$
# 
# 
# - $W(t_i)$ proceso de Wiener en el tiempo $t_i$
# - $W(t_{i-1})$ proceso de Wiener en el tiempo $t_{i-1}$
# - $\Delta t$ tamaño del paso
# - $Z_i$ es un número aleatorio extraído de una distribución normal estándar
# 
# Este proceso se repite para cada $t_i$ en el rango de $1$ a $n$, generando la trayectoria completa del proceso de Wiener. La salida es una secuencia de valores de $W(t_i)$ que representa una realización específica del proceso de Wiener en el intervalo de tiempo $[0, T]$ con $n$ pasos.

# In[3]:


n = 10
T = 1
dt = T / n
W = np.zeros(n+1)

# Simulación del proceso de Wiener
for i in range(1, n+1):
    dW = np.sqrt(dt) * np.random.normal()  # Incremento de Wiener
    W[i] = W[i-1] + dW

print(W)


# ## Simulación de n=10

# ### I(t) como variable aleatoria

# ### I(t)=$\sum_{i=1}^{n} W(t_i)[W(t_i) - W(t_{i-1})] $

# 
# 1. Inicialización de \(I\):
#    - $I = 0$
# 
# 2. Iteración sobre \(i\) desde 0 hasta \(n-1\):
#    - $I = I + W[i] \cdot (W[i] - W[i+1])$
#    - $It[i] = I$
# 
# Siendo el proceso dado por la variable aleatoria
# 
# 
# $I(t) = \sum_{i=1}^{n} W(t_i) \cdot [W(t_i) - W(t_{i-1})]$
# 
# Donde:
# - $I(t)$ es el proceso de Winer que calculamos dado $W(t)$
# - $W(t_i)$ son los valores del proceso de Wiener en diferentes puntos de tiempo $t_i$.
# - La suma se realiza sobre $i$ desde $1$ hasta $n$.
# 
# La variable aleatoria $I(t)$ se calcula como la suma de productos de incrementos sucesivos del proceso de Wiener en cada paso de tiempo. Lo cual refleja la acumulación de la diferencia entre los valores del proceso de Wiener en cada paso. El resultado `It`  contiene los valores acumulados de $I(t)$ en cada punto de tiempo.

# In[10]:


import numpy as np

def simulate_wiener(n, T):
    dt = T / n
    W = np.cumsum(np.sqrt(dt) * np.random.normal(size=n))
    return W

def simulate_I(W):
    I = np.cumsum(W[:-1] * np.diff(W))
    return np.concatenate(([0], I))

# Parámetros
n = 10
T = 1

# Simulación del proceso de Wiener
np.random.seed(25)
W = simulate_wiener(n, T)

# Simulación de la variable aleatoria I(t)
It = simulate_I(W)

print("W:", W)
print("It:", It)


# Ambas funciones están diseñadas para trabajar juntas en la simulación de un proceso de Wiener y la variable aleatoria $I(t)$ asociada. La función `simulate_I` toma la realización del proceso de Wiener `W` como entrada y calcula la variable aleatoria $I(t)$ basándose en esa realización. 
# 

# ### Graficas 

# In[16]:


# Simulación del proceso de Wiener
W = simulate_wiener(n, T)

#Recordemos que la variable aleatoria I(t)
#esta en funcion del movimiento de Wiener 

# Simulación de la variable aleatoria I(t)
It = simulate_I(W)

# Gráfico del proceso de Wiener
plt.plot(np.linspace(0, T, n), W, color='red', linestyle='-', marker='o')
plt.xlabel('Tiempo')
plt.ylabel('W(t)')
plt.title('Proceso de Wiener')
plt.show()

# Gráfico de la variable aleatoria I(t)
plt.plot(np.linspace(0, T-dt, n), It, color='purple', linestyle='-', marker='o')
plt.xlabel('Tiempo')
plt.ylabel('I(t)')
plt.title('Variable Aleatoria I(t)')
plt.show()


# ## Simulación de n=50

# In[17]:


# Parámetros
n = 50
T = 1

# Simulación del proceso de Wiener
np.random.seed(25)
W = simulate_wiener(n, T)

# Simulación de la variable aleatoria I(t)
It = simulate_I(W)

print("W:", W)
print("It:", It)


# In[25]:


# Simulación del proceso de Wiener
W = simulate_wiener(n, T)

#Recordemos que la variable aleatoria I(t)
#esta en funcion del movimiento de Wiener 

# Simulación de la variable aleatoria I(t)
It = simulate_I(W)

# Gráfico del proceso de Wiener
plt.plot(np.linspace(0, T, n), W, color='black', linestyle='-', marker='*')
plt.xlabel('Tiempo')
plt.ylabel('W(t)')
plt.title('Proceso de Wiener')
plt.show()

# Gráfico de la variable aleatoria I(t)
plt.plot(np.linspace(0, T-dt, n), It, color='blue', linestyle='-', marker='+')
plt.xlabel('Tiempo')
plt.ylabel('I(t)')
plt.title('Variable Aleatoria I(t)')
plt.show()


# ## Simulación de n=100

# In[19]:


# Parámetros
n = 100
T = 1

# Simulación del proceso de Wiener
np.random.seed(25)
W = simulate_wiener(n, T)

# Simulación de la variable aleatoria I(t)
It = simulate_I(W)

print("W:", W)
print("It:", It)


# In[26]:


# Simulación del proceso de Wiener
W = simulate_wiener(n, T)

#Recordemos que la variable aleatoria I(t)
#esta en funcion del movimiento de Wiener 

# Simulación de la variable aleatoria I(t)
It = simulate_I(W)

# Gráfico del proceso de Wiener
plt.plot(np.linspace(0, T, n), W, color='green', linestyle='-', marker='p')
plt.xlabel('Tiempo')
plt.ylabel('W(t)')
plt.title('Proceso de Wiener')
plt.show()

# Gráfico de la variable aleatoria I(t)
plt.plot(np.linspace(0, T-dt, n), It, color='yellow', linestyle='-', marker='s')
plt.xlabel('Tiempo')
plt.ylabel('I(t)')
plt.title('Variable Aleatoria I(t)')
plt.show()


# ## Simulación de $X(t,w(t))$

# ####  $X(t,w(t))=\sum_{i=1}^{n} W(t_{i-1,i})I_{[t_{i-1},t_i)} $

# In[51]:


def SimWeiner(n, T):
    dt = T / n
    t_values = np.arange(0, T, dt)
    dW = np.sqrt(dt) * np.random.normal(0, 1, n)
    W = np.cumsum(dW)

    return t_values, W


# 
# 1. **SimWeiner(n, T):**
#    Esta función simula un proceso de Wiener $W(t)$ discretizado en $n$ pasos de tiempo hasta un tiempo total $T$.
#    
#    - $dt$ es el tamaño del paso de tiempo, calculado como $dt = \frac{T}{n}$.
#    - $t\_values$ es una secuencia de tiempos discretos $(0, dt, 2*dt, \ldots, T-dt)$.
#    - $dW$ es el incremento de Wiener en cada paso de tiempo, donde $dW_i \sim \mathcal{N}(0, dt)$.
#    - $W$ es el proceso de Wiener acumulado, calculado como la suma acumulativa de los incrementos de Wiener.
# 
#    $\Delta W_i = \sqrt{dt} \cdot \mathcal{N}(0, 1)$
#    y
#    $W(t_i) = W(t_{i-1}) + \Delta W_i$
# 
# 

# In[52]:


def SimX(t_values, W):
    n = len(t_values)
    X = np.zeros(n - 1)

    for i in range(1, n):
        X[i - 1] = W[i] * (t_values[i] - t_values[i - 1])

    return X


# SimX(t_values, W):
#    Esta función simula el proceso $X(t, W(t))$ basado en los valores de $t$ y $W$ generados por la función anterior.
#    - $X$ es un proceso definido como $X(t, W(t)) = \sum_{i=1}^{n} W(t_{i-1, i}) I_{[t_{i-1}, t_i)}$, donde $I_{[a, b)}$ es la función indicadora que toma el valor 1 si $a \leq t < b$ o  0 en el caso contrario.
# 
#    $X(t_i, W(t_i)) = W(t_{i-1}) \cdot (t_i - t_{i-1})$
# 

# In[53]:



def plotPP(t_values, W, X):
    plt.plot(t_values, W, label='Proceso de Wiener ')
    plt.step(t_values[1:], X, where='post', label='X(t, W(t))')
    plt.xlabel('Tiempo (t)')
    plt.legend()
    plt.show()


# ## Simulación para n=10

# In[48]:


n = 10 # Número de pasos
T= 1.0   
t_vals, W_vals = SimWeiner(n, T)
X_vals = SimX(t_vals, W_vals)
plotPP(t_vals, W_vals, X_vals)


# ## Simulación para n=50

# In[43]:


n = 50  # Número de pasos
T= 1.0  
t_vals, W_vals = SimWeiner(n, T)
X_vals = SimX(t_vals, W_vals)
plotPP(t_vals, W_vals, X_vals)


# ## Simulación para n=100

# In[44]:


n = 100  # Número de pasos
T= 1.0  
t_vals, W_vals = SimWeiner(n, T)
X_vals = SimX(t_vals, W_vals)
plotPP(t_vals, W_vals, X_vals)

