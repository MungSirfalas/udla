# %% [markdown]
# # Introducción a la Estadística Descriptiva
# ## Curso de Maestría en Ciencia de Datos

# %% [markdown]
# ### Introducción
# 
# La estadística descriptiva es una rama fundamental de la estadística que se ocupa de la recolección, organización, análisis y presentación de datos. En esta era de información masiva, la capacidad de comprender y describir datos de manera efectiva se ha vuelto una habilidad esencial para cualquier científico de datos.
# 
# A diferencia de la estadística inferencial, que busca hacer predicciones y generalizaciones sobre una población basándose en una muestra, la estadística descriptiva se centra en describir y resumir las características principales de un conjunto de datos.
# 
# En el mundo actual, donde la toma de decisiones basada en datos se ha convertido en un estándar en prácticamente todas las industrias, la capacidad de entender y aplicar conceptos estadísticos es más importante que nunca.

# %%
# Importación de librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las visualizaciones
plt.style.use('seaborn')
sns.set_palette("husl")

# %% [markdown]
# ### 1. Variables Cualitativas y Cuantitativas
# 
# El primer paso para cualquier análisis estadístico es comprender la naturaleza de los datos con los que estamos trabajando. La clasificación de variables en cualitativas y cuantitativas es fundamental porque determina qué tipos de análisis y visualizaciones son apropiados para nuestros datos.
# 
# #### 1.1 Variables Cualitativas (Categóricas)
# 
# * **Nominales:**
#   - No tienen un orden natural
#   - Ejemplos: género, color de ojos, nacionalidad
#   - Operaciones permitidas: igual a (=) o diferente de (≠)
#   - Medidas descriptivas: moda, frecuencias
# 
# * **Ordinales:**
#   - Tienen un orden o ranking natural
#   - Ejemplos: nivel educativo, satisfacción del cliente
#   - Operaciones permitidas: mayor que (>), menor que (<), igual a (=)
#   - Medidas descriptivas: mediana, moda, frecuencias
# 
# #### 1.2 Variables Cuantitativas (Numéricas)
# 
# * **Discretas:**
#   - Toman valores enteros o contables
#   - Ejemplos: número de hijos, número de estudiantes
#   - Medidas descriptivas: media, mediana, moda, varianza
# 
# * **Continuas:**
#   - Pueden tomar cualquier valor dentro de un intervalo
#   - Ejemplos: altura, peso, temperatura
#   - Medidas descriptivas: media, mediana, moda, varianza, desviación estándar

# %%
# Crear datos de ejemplo
np.random.seed(42)
n_students = 200
data = {
    'edad': np.random.normal(20, 2, n_students),  # Cuantitativa continua
    'semestre': np.random.randint(1, 11, n_students),  # Cuantitativa discreta
    'genero': np.random.choice(['M', 'F'], n_students),  # Cualitativa nominal
    'nivel_satisfaccion': np.random.choice(['Bajo', 'Medio', 'Alto'], n_students),  # Cualitativa ordinal
    'num_cursos': np.random.poisson(3, n_students)  # Cuantitativa discreta
}

df = pd.DataFrame(data)

# Mostrar información sobre las variables
print("Análisis de Variables:")
print("\nVariables Cuantitativas:")
print(df.describe())
    
print("\nVariables Cualitativas:")
for col in ['genero', 'nivel_satisfaccion']:
    print(f"\nDistribución de {col}:")
    print(df[col].value_counts())

# %% [markdown]
# ### 2. Histogramas y Distribuciones
# 
# La visualización de datos es una herramienta poderosa en estadística descriptiva. Los histogramas y otras representaciones gráficas nos permiten "ver" patrones en los datos que podrían no ser evidentes en una tabla de números.
# 
# #### 2.1 Visualización de Variables Cualitativas
# * Gráficos de barras
# * Gráficos circulares
# 
# #### 2.2 Visualización de Variables Cuantitativas
# * Histogramas
# * Polígonos de frecuencia

# %%
# Crear visualizaciones
fig = plt.figure(figsize=(15, 10))

# Gráfico de barras para género
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='genero')
plt.title('Distribución por Género')

# Gráfico de barras para nivel de satisfacción
plt.subplot(2, 2, 2)
sns.countplot(data=df, x='nivel_satisfaccion')
plt.title('Distribución por Nivel de Satisfacción')

# Histograma para edad
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='edad', kde=True)
plt.title('Distribución de Edades')

# Histograma para número de cursos
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='num_cursos', kde=True)
plt.title('Distribución de Número de Cursos')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Función de Probabilidad de Masa (PMF)
# 
# La función de probabilidad de masa es un concepto fundamental en la teoría de probabilidad y estadística, especialmente cuando trabajamos con variables aleatorias discretas.
# 
# Características principales:
# * Solo está definida para valores discretos
# * P(X = x) ≥ 0 para todo x
# * La suma de todas las probabilidades es 1

# %%
# Visualizar PMF
# Ejemplo: Lanzamiento de un dado
x = np.arange(1, 7)
pmf = np.ones(6) / 6  # Probabilidad uniforme para cada resultado

plt.figure(figsize=(10, 6))
plt.bar(x, pmf)
plt.title('PMF de un Dado Justo')
plt.xlabel('Resultado')
plt.ylabel('Probabilidad')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.show()

# Ejemplo con datos discretos reales
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='num_cursos', stat='probability')
plt.title('PMF Empírica del Número de Cursos')
plt.xlabel('Número de Cursos')
plt.ylabel('Probabilidad')
plt.show()

# %% [markdown]
# ### 4. Función de Densidad de Probabilidad (PDF)
# 
# La función de densidad de probabilidad es el equivalente continuo de la PMF. Mientras que la PMF nos da probabilidades exactas para valores discretos, la PDF nos da la "densidad" de probabilidad en cada punto de un continuo.
# 
# Características principales:
# * Es una función continua
# * El área bajo la curva es 1
# * f(x) ≥ 0 para todo x
# * P(a ≤ X ≤ b) = ∫[a,b] f(x)dx

# %%
# Visualizar PDF
# Ejemplo teórico: Distribución Normal
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', label='PDF Teórica')
plt.title('PDF de una Distribución Normal Estándar')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Ejemplo con datos reales
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='edad')
plt.title('PDF Empírica de las Edades')
plt.xlabel('Edad')
plt.ylabel('Densidad')
plt.show()

# %% [markdown]
# ### 5. Función de Distribución Acumulada (CDF)
# 
# La función de distribución acumulada es una herramienta que nos permite calcular probabilidades para rangos de valores. Es especialmente útil cuando queremos saber la probabilidad de que una variable aleatoria sea menor o igual a un valor específico.
# 
# Propiedades:
# 1. F(x) es monótona creciente
# 2. lim(x→-∞) F(x) = 0
# 3. lim(x→∞) F(x) = 1
# 4. P(a < X ≤ b) = F(b) - F(a)

# %%
# Visualizar CDF
# Ejemplo teórico: CDF Normal
x = np.linspace(-4, 4, 1000)
cdf = stats.norm.cdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, cdf, 'r-', label='CDF Teórica')
plt.title('CDF de una Distribución Normal Estándar')
plt.xlabel('x')
plt.ylabel('Probabilidad Acumulada')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Ejemplo con datos reales
plt.figure(figsize=(10, 6))
sns.ecdfplot(data=df, x='edad')
plt.title('CDF Empírica de las Edades')
plt.xlabel('Edad')
plt.ylabel('Probabilidad Acumulada')
plt.show()

# %% [markdown]
# ### 6. Probabilidad en Estadística
# 
# La probabilidad es el lenguaje matemático de la incertidumbre. Estudiaremos:
# 
# * **Definiciones de Probabilidad:**
#   - Clásica (Laplace)
#   - Frecuentista
#   - Axiomática (Kolmogorov)
# 
# * **Reglas Básicas:**
#   - Regla de la Suma
#   - Regla del Producto
#   - Probabilidad Condicional

# %%
# Visualizar probabilidades
# Probabilidad conjunta
cont_table = pd.crosstab(df['genero'], df['nivel_satisfaccion'], normalize='all')

plt.figure(figsize=(10, 6))
sns.heatmap(cont_table, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Probabilidades Conjuntas: Género vs Nivel de Satisfacción')
plt.show()

# Probabilidades condicionales
cond_prob = pd.crosstab(df['genero'], df['nivel_satisfaccion'], normalize='index')

plt.figure(figsize=(10, 6))
sns.heatmap(cond_prob, annot=True, fmt='.3f', cmap='YlGnBu')
plt.title('Probabilidades Condicionales: P(Nivel de Satisfacción | Género)')
plt.show()

# %% [markdown]
# ### Conclusión
# 
# La estadística descriptiva es mucho más que un conjunto de técnicas para resumir datos. Es un lenguaje fundamental para comunicar información cuantitativa y una herramienta esencial para la toma de decisiones basada en datos.
# 
# Los conceptos presentados en este material forman la base para análisis más avanzados y son esenciales para cualquier científico de datos.

# %%
# Guardar los datos de ejemplo
df.to_csv('datos_ejemplo_estadistica.csv', index=False)
print("Datos guardados en 'datos_ejemplo_estadistica.csv'")
