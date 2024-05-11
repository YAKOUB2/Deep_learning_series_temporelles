import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller


class Analyse:

    date = "Date Time"

    def __init__(self, data):
        self.data = data
        
    def summary_statistics(self):
        stats_df = self.data.describe()
        stats_df = stats_df.transpose()  # Transpose pour avoir les variables en lignes
        return stats_df
    
    def missing_values(self):
        missing_counts = self.data.isnull().sum()
        missing_df = pd.DataFrame({'Missing Values': missing_counts})
        return missing_df
    
    def correlation_matrix(self):
        columns = ['p (mbar)', 'T (degC)', 'rh (%)', 
                'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 
                'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 
                'wd (deg)']
        # Calculer la matrice de corrélation
        corr_matrix = self.data[columns].corr()

        # Visualiser la matrice de corrélation
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Matrice de corrélation')
        plt.show()

    def distribution_plot(self, column):
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    
    def boxplot(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data)
        plt.title('Box plot des variables')
        plt.xticks(rotation=45)
        plt.show()

    def time_series(self, column):
        plt.figure(figsize=(20, 6))
        plt.plot(self.data[Analyse.date], self.data[column])
        plt.xlabel(Analyse.date)
        plt.ylabel(column)
        plt.show()
    
    def test_stationarity(self, column):

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(self.data[column], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        
        # Check if the column is stationary:
        if dfoutput['p-value'] < 0.05:  # Assuming 0.05 as the significance level
            print(f"La variable {column} est stationnaire.")
        else:
            print(f"La variable {column} n'est pas stationnaire.")
            
        return dfoutput


    
    def display(self):
        return self.data.head()
    
    
    def plot_time_series(self, indicators=['T (degC)', 'rh (%)', 'wv (m/s)'], aggregate=True):

        if aggregate:
            freq_map = {'day': 'D', 'week': 'W', 'month': 'M', 'year': 'Y'}
            for frequency, freq_label in freq_map.items():
                self._plot(indicators, frequency, freq_label)
        else:
            self._plot(indicators)

    def _plot(self, indicators, frequency=None, freq_label=None):

        if frequency:
            df_agg = self.data.resample(freq_label, on='Date Time').mean()
            title = f"Séries temporelles ({frequency})"
        else:
            df_agg = self.data
            title = "Séries temporelles"

        # Tracé des séries temporelles sur un même graphique avec légende
        fig = px.line(title=title)
        for indicator in indicators:
            fig.add_scatter(x=df_agg.index, y=df_agg[indicator], mode='lines', name=indicator)
        
        fig.show()


    def plot_time_series_separated(self, indicators=['T (degC)', 'rh (%)', 'wv (m/s)'], frequency='day', aggregate=True):

        if aggregate:
            freq_map = {'day': 'D', 'week': 'W', 'month': 'M', 'year': 'Y'}
            self.data = self.data.resample(freq_map.get(frequency, 'D'), on='Date Time').mean()

        # Nombre de graphiques et calcul de nombre de lignes nécessaires
        num_graphs = len(indicators)
        rows = (num_graphs + 2) // 3  # Arrondi au nombre supérieur pour inclure tous les graphiques
        
        # Création de la figure avec des sous-graphiques
        fig = make_subplots(rows=rows, cols=3, subplot_titles=[f"{indicator} ({frequency})" for indicator in indicators])

        # Ajout des graphiques à la figure
        for index, indicator in enumerate(indicators):
            row = index // 3 + 1
            col = index % 3 + 1
            fig.add_trace(
                go.Scatter(x=self.data.index, y=self.data[indicator], mode='lines', name=indicator),
                row=row, col=col
            )

        # Mise à jour de la mise en page et affichage de la figure
        fig.update_layout(height=300 * rows, width=1200, title_text="Séries temporelles des indicateurs météo")
        fig.show()

    def plot_density(self):
        quantitative_vars = self.data.select_dtypes(include=['float64', 'int64'])

        # Calcul du nombre de sous-graphiques nécessaires
        num_vars = len(quantitative_vars.columns)
        num_rows = (num_vars + 1) // 2  # Arrondi au nombre supérieur pour le nombre de lignes
        num_cols = min(2, num_vars)  # Maximum de 2 colonnes

        # Tracé des distributions de densité
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(quantitative_vars.columns):
            plt.subplot(num_rows, num_cols, i+1)
            sns.kdeplot(data=self.data, x=col, fill=True)
            plt.title(f'Distribution de densité de {col}')
        plt.tight_layout()
        plt.show()
        
    def plot_boxplots(self):
        # Sélection des variables quantitatives
        quantitative_vars = self.data.select_dtypes(include=['float64', 'int64'])

        # Calcul du nombre de sous-graphiques nécessaires
        num_vars = len(quantitative_vars.columns)
        num_rows = (num_vars + 1) // 2  # Arrondi au nombre supérieur pour le nombre de lignes
        num_cols = min(2, num_vars)  # Maximum de 2 colonnes

        # Tracé des boxplots
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(quantitative_vars.columns):
            plt.subplot(num_rows, num_cols, i+1)
            sns.boxplot(data=self.data, y=col)
            plt.title(f'Boxplot de {col}')
        plt.tight_layout()
        plt.show()
        
        
    def plot_temperature_hourly_boxplot(self):

        # Tracer les boxplots de la température par heure sur toute la période
        fig = px.box(self.data, x=self.data['Date Time'].dt.hour, y='Tdew (degC)', title="Boxplots de la température par heure sur toute la période",
                    labels={'Date Time': 'Heure', 'Tdew (degC)': 'Température (degC)'})
        fig.update_xaxes(title="Heure", tickvals=list(range(24)), ticktext=[str(i) + 'h' for i in range(24)])
        fig.show()
        
    def plot_temperature_monthly_boxplot(self):

        # Tracer les boxplots de la température par mois sur toute la période
        fig = px.box(self.data, x=self.data['Date Time'].dt.month, y='Tdew (degC)', title="Boxplots de la température par mois sur toute la période",
                    labels={'Date Time': 'Mois', 'Tdew (degC)': 'Température (degC)'})
        fig.update_xaxes(title="Mois", tickvals=list(range(1, 13)), ticktext=[str(i) for i in range(1, 13)])
        fig.show()

