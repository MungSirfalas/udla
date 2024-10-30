{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Estadística Descriptiva con Python\n",
    "## Curso de Introducción a la Estadística - Maestría en Ciencia de Datos\n",
    "\n",
    "En este notebook, exploraremos los conceptos fundamentales de la estadística descriptiva utilizando Python y sus principales bibliotecas para análisis de datos y visualización."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Importación de bibliotecas necesarias\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy import stats\n",
    "\n",
    "# Configuración de estilo para las visualizaciones\n",
    "plt.style.use('seaborn')\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Variables Cualitativas y Cuantitativas\n",
    "\n",
    "### 1.1 Variables Cualitativas (Categóricas)\n",
    "Vamos a crear un conjunto de datos de ejemplo para mostrar el análisis de variables cualitativas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Crear datos de ejemplo para variables cualitativas\n",
    "np.random.seed(42)\n",
    "n_samples = 200\n",
    "\n",
    "# Variables nominales\n",
    "colores = np.random.choice(['Rojo', 'Azul', 'Verde', 'Amarillo'], n_samples, p=[0.3, 0.3, 0.2, 0.2])\n",
    "\n",
    "# Variables ordinales\n",
    "niveles_educativos = np.random.choice(['Primaria', 'Secundaria', 'Universidad', 'Posgrado'], \n",
    "                                      n_samples, p=[0.2, 0.3, 0.35, 0.15])\n",
    "\n",
    "# Crear DataFrame\n",
    "df_cualitativo = pd.DataFrame({\n",
    "    'Color_Preferido': colores,\n",
    "    'Nivel_Educativo': niveles_educativos\n",
    "})\n",
    "\n",
    "# Mostrar las primeras filas\n",
    "print(\"Muestra de datos cualitativos:\")\n",
    "print(df_cualitativo.head())\n",
    "\n",
    "# Análisis de frecuencias\n",
    "print(\"\\nTabla de frecuencias para Color Preferido:\")\n",
    "print(df_cualitativo['Color_Preferido'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualización de variables cualitativas\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
    "\n",
    "# Gráfico de barras para Color Preferido\n",
    "sns.countplot(data=df_cualitativo, x='Color_Preferido', ax=ax1)\n",
    "ax1.set_title('Distribución de Colores Preferidos')\n",
    "ax1.set_ylabel('Frecuencia')\n",
    "plt.xticks(rotation=45)\n",
    "\n",
    "# Gráfico de barras para Nivel Educativo\n",
    "sns.countplot(data=df_cualitativo, x='Nivel_Educativo', ax=ax2)\n",
    "ax2.set_title('Distribución de Niveles Educativos')\n",
    "ax2.set_ylabel('Frecuencia')\n",
    "plt.xticks(rotation=45)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.2 Variables Cuantitativas\n",
    "Ahora generaremos datos para mostrar el análisis de variables cuantitativas discretas y continuas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Crear datos de ejemplo para variables cuantitativas\n",
    "# Variable discreta: número de hijos\n",
    "num_hijos = np.random.poisson(lam=2, size=n_samples)\n",
    "\n",
    "# Variables continuas\n",
    "altura = np.random.normal(170, 10, n_samples)  # media 170, desv. est. 10\n",
    "peso = np.random.normal(70, 15, n_samples)     # media 70, desv. est. 15\n",
    "\n",
    "# Crear DataFrame\n",
    "df_cuantitativo = pd.DataFrame({\n",
    "    'Num_Hijos': num_hijos,\n",
    "    'Altura': altura,\n",
    "    'Peso': peso\n",
    "})\n",
    "\n",
    "# Estadísticas descriptivas\n",
    "print(\"Estadísticas descriptivas de variables cuantitativas:\")\n",
    "print(df_cuantitativo.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualización de variables cuantitativas\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
    "\n",
    "# Histograma de altura\n",
    "sns.histplot(data=df_cuantitativo, x='Altura', bins=30, ax=axes[0,0])\n",
    "axes[0,0].set_title('Distribución de Altura')\n",
    "\n",
    "# Histograma de peso\n",
    "sns.histplot(data=df_cuantitativo, x='Peso', bins=30, ax=axes[0,1])\n",
    "axes[0,1].set_title('Distribución de Peso')\n",
    "\n",
    "# Gráfico de barras para número de hijos (variable discreta)\n",
    "sns.countplot(data=df_cuantitativo, x='Num_Hijos', ax=axes[1,0])\n",
    "axes[1,0].set_title('Distribución de Número de Hijos')\n",
    "\n",
    "# Boxplot de variables continuas\n",
    "df_cuantitativo.boxplot(column=['Altura', 'Peso'], ax=axes[1,1])\n",
    "axes[1,1].set_title('Boxplots de Altura y Peso')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Función de Probabilidad de Masa (PMF)\n",
    "Vamos a ilustrar la PMF usando el ejemplo del número de hijos (variable discreta)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Calcular PMF para número de hijos\n",
    "pmf_hijos = df_cuantitativo['Num_Hijos'].value_counts(normalize=True).sort_index()\n",
    "\n",
    "# Visualizar PMF\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.bar(pmf_hijos.index, pmf_hijos.values)\n",
    "plt.title('Función de Probabilidad de Masa - Número de Hijos')\n",
    "plt.xlabel('Número de Hijos')\n",
    "plt.ylabel('Probabilidad')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()\n",
    "\n",
    "print(\"Probabilidades para cada valor:\")\n",
    "print(pmf_hijos)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Función de Densidad de Probabilidad (PDF)\n",
    "Ilustraremos la PDF usando la altura (variable continua) y compararemos con una distribución normal teórica."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Calcular PDF empírica y teórica\n",
    "altura_mean = df_cuantitativo['Altura'].mean()\n",
    "altura_std = df_cuantitativo['Altura'].std()\n",
    "\n",
    "# Crear puntos para la curva teórica\n",
    "x = np.linspace(altura_mean - 4*altura_std, altura_mean + 4*altura_std, 100)\n",
    "pdf_teorica = stats.norm.pdf(x, altura_mean, altura_std)\n",
    "\n",
    "# Visualizar PDF\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(data=df_cuantitativo, x='Altura', stat='density', bins=30, alpha=0.5)\n",
    "plt.plot(x, pdf_teorica, 'r-', lw=2, label='PDF teórica (Normal)')\n",
    "plt.title('Función de Densidad de Probabilidad - Altura')\n",
    "plt.xlabel('Altura (cm)')\n",
    "plt.ylabel('Densidad')\n",
    "plt.legend()\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Función de Distribución Acumulada (CDF)\n",
    "Mostraremos la CDF tanto para variables discretas como continuas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# CDF para variable discreta (número de hijos)\n",
    "cdf_hijos = df_cuantitativo['Num_Hijos'].value_counts(normalize=True).sort_index().cumsum()\n",
    "\n",
    "# CDF para variable continua (altura)\n",
    "altura_sorted = np.sort(df_cuantitativo['Altura'])\n",
    "p = np.arange(len(altura_sorted)) / (len(altura_sorted) - 1)\n",
    "\n",
    "# Visualizar CDFs\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
    "\n",
    "# CDF discreta\n",
    "ax1.step(cdf_hijos.index, cdf_hijos.values, where='post')\n",
    "ax1.set_title('CDF - Número de Hijos')\n",
    "ax1.set_xlabel('Número de Hijos')\n",
    "ax1.set_ylabel('Probabilidad Acumulada')\n",
    "ax1.grid(True, alpha=0.3)\n",
    "\n",
    "# CDF continua\n",
    "ax2.plot(altura_sorted, p)\n",
    "ax2.set_title('CDF - Altura')\n",
    "ax2.set_xlabel('Altura (cm)')\n",
    "ax2.set_ylabel('Probabilidad Acumulada')\n",
    "ax2.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Probabilidad en Estadística\n",
    "Vamos a ilustrar algunos conceptos básicos de probabilidad usando nuestros datos."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Calcular algunas probabilidades de ejemplo\n",
    "\n",
    "# Probabilidad de tener más de 2 hijos\n",
    "p_mas_2_hijos = (df_cuantitativo['Num_Hijos'] > 2).mean()\n",
    "\n",
    "# Probabilidad de tener altura entre 160 y 180 cm\n",
    "p_altura_rango = ((df_cuantitativo['Altura'] >= 160) & \n",
    "                  (df_cuantitativo['Altura'] <= 180)).mean()\n",
    "\n",
    "print(f\"Probabilidad de tener más de 2 hijos: {p_mas_2_hijos:.3f}\")\n",
    "print(f\"Probabilidad de tener altura entre 160 y 180 cm: {p_altura_rango:.3f}\")\n",
    "\n",
    "# Visualizar estas probabilidades\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(data=df_cuantitativo, x='Altura', bins=30)\n",
    "plt.axvline(x=160, color='r', linestyle='--', label='Límite inferior (160 cm)')\n",
    "plt.axvline(x=180, color='r', linestyle='--', label='Límite superior (180 cm)')\n",
    "plt.title('Distribución de Altura con Rango de Interés')\n",
    "plt.xlabel('Altura (cm)')\n",
    "plt.ylabel('Frecuencia')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
 ]
}
