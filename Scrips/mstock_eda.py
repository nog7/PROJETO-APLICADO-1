# %%
# Importação de bibliotecas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Análise Exploratória de Dados",
    page_icon=":bar_chart:",  # Emoji de gráfico para análise de dados
    layout="wide",  # ou "centered"
)


df = pd.read_csv('C:\\Users\\euller.nogueira\\Documents\\Atividades\\PROJETO APLICADO I\\Notebooks\\Dataset\\Microsoft_Stock.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


def load_css(file_name):
    with open(file_name) as f:
        return f.read()

st.markdown(f'<style>{load_css("style.css")}</style>', unsafe_allow_html=True)

# Adicionar uma sidebar para navegação
st.sidebar.markdown('<h1 style="text-align: center;">Navegação</h1>', unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'Informações dos Dados'

# Adiciona botões personalizados na barra lateral
if st.sidebar.button('Informações dos Dados', key='info'):
    st.session_state.page = 'Informações dos Dados'
if st.sidebar.button('Análise Exploratória dos Dados', key='analysis'):
    st.session_state.page = 'Análise Exploratória dos Dados'

# Seções baseadas na seleção da sidebar
if st.session_state.page == 'Informações dos Dados':

    st.header('Informações dos Dados')
    
    st.subheader('Quantidades de Linhas e Colunas:')
    num_rows, num_cols = df.shape
    st.write(f'O dataset possui {num_rows} linhas e {num_cols} colunas.')

    st.subheader('Head:')
    st.dataframe(df.head()) 
    st.subheader('Tail:')
    st.dataframe(df.tail())   
    
    st.subheader('Estatísticas descritivas:')
    st.write(df.describe())
    
    st.subheader('Verificação de valores ausentes:')
    st.write(df.isnull().sum())

elif st.session_state.page == 'Análise Exploratória dos Dados':
    st.header('Análise Exploratória dos Dados')

    # Criar uma lista suspensa para selecionar o gráfico

    # Nova opção para a análise
    analysis_option = st.selectbox(
        'Escolha a análise a ser exibida:',
        ['Gráficos de Séries Temporais', 'Preço Mais Alto e Mais Baixo', 'Preço de Fechamento e Média Móvel', 'Mudança Diária no Preço', 'Previsão com SARIMAX']
    )
    
    if analysis_option == 'Gráficos de Séries Temporais':
        st.subheader('Gráficos de Séries Temporais')

        columns = ['Open', 'Close', 'High', 'Low']
        colors = ['blue', 'purple', 'green', 'red']

        # Criar subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=columns)

        for i, (col, color) in enumerate(zip(columns, colors)):
            row = i // 2 + 1
            col_num = i % 2 + 1
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode='lines',
                name=col,
                line=dict(color=color)
            ), row=row, col=col_num)

        fig.update_layout(
            title='Séries Temporais das Ações da Microsoft',
            height=800,
            width=1000
        )

        st.plotly_chart(fig)

    # Seção de Preço Mais Alto e Mais Baixo
    elif analysis_option == 'Preço Mais Alto e Mais Baixo':
        st.subheader('Preço Mais Alto e Mais Baixo')

        def plot_combined_series(df, columns, titles, colors):
            fig = go.Figure()
            for column, title, color in zip(columns, titles, colors):
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[column],
                    mode='lines',
                    name=title,
                    line=dict(color=color, width=1.5)  
                ))
            fig.update_layout(
                title='Série Temporal: Maior e Menor Preço',
                xaxis_title='Data',
                yaxis_title='Valores',
                xaxis=dict(tickangle=45),
                legend_title='Preços',
                height=600,
                width=900
            )
            return fig

        # Plotar os mais alto e mais baixo em um único gráfico
        combined_fig = plot_combined_series(df, ['High', 'Low'], ['Preço Mais Alto', 'Preço Mais Baixo'], ['green', 'red'])
        st.plotly_chart(combined_fig)

    # Seção de Preço de Fechamento e Média Móvel
    elif analysis_option == 'Preço de Fechamento e Média Móvel':
        st.subheader('Preço de Fechamento e Média Móvel')

        df['Moving Average'] = df['Close'].rolling(window=180).mean()

        # plotar o preço de fechamento e a média móvel usando Plotly
        def plot_moving_average(df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                mode='lines',
                name='Preço de Fechamento',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Moving Average'],
                mode='lines',
                name='Média Móvel Semestral',
                line=dict(color='orange')
            ))
            fig.update_layout(
                title='Preço de Fechamento e Média Móvel',
                xaxis_title='Data',
                yaxis_title='Preço de Fechamento (USD)',
                xaxis=dict(tickangle=45),
                legend_title='Legenda',
                height=600,
                width=900
            )
            return fig

        moving_avg_fig = plot_moving_average(df)
        st.plotly_chart(moving_avg_fig)

    # Seção de Mudança Diária no Preço
    elif analysis_option == 'Mudança Diária no Preço':
        st.subheader('Mudança Diária no Preço')

        # Adicionar a coluna de mudança diária
        df['Mudança Diária'] = df['Close'] - df['Open']

        # Função para plotar a mudança diária no preço usando Plotly
        def plot_daily_change(df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Mudança Diária'],
                mode='lines',
                name='Mudança Diária',
                line=dict(color='orange')
            ))
            fig.update_layout(
                title='Mudança Diária no Preço',
                xaxis_title='Data',
                yaxis_title='Mudança Diária (USD)',
                xaxis=dict(tickangle=45),
                legend_title='Legenda',
                height=600,
                width=900
            )
            return fig

        # Plotar o gráfico
        daily_change_fig = plot_daily_change(df)
        st.plotly_chart(daily_change_fig)


    # Previsão com SARIMAX
    if analysis_option == 'Previsão com SARIMAX':
        st.subheader('Previsão com SARIMAX')

        closing_prices = df['Close']

        # parâmetros do modelo SARIMAX 
        p, d, q = 1, 1, 1  
        P, D, Q, s = 1, 1, 1, 12 
        
        model = SARIMAX(closing_prices, order=(p, d, q), seasonal_order=(P, D, Q, s))
        sarimax_model = model.fit(disp=False)

        # passos de previsão
        passos_previsao = 240

        forecast = sarimax_model.get_forecast(steps=passos_previsao)
        forecast_mean = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()

        forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=passos_previsao)

        # dados históricos de fechamento e a previsão
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=closing_prices, mode='lines', name='Preço de Fechamento',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_mean, mode='lines', name='Previsão SARIMAX',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_conf_int['lower Close'], mode='lines', name='Limite Inferior',
            line=dict(color='rgba(255, 0, 0, 0.3)')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_conf_int['upper Close'], mode='lines', name='Limite Superior',
            line=dict(color='rgba(255, 0, 0, 0.3)')
        ))
        fig.update_layout(
            title='Previsão do Preço de Fechamento com SARIMAX',
            xaxis_title='Data',
            yaxis_title='Preço de Fechamento (USD)',
            legend_title='Legenda',
            height=600,
            width=900
        )

        st.plotly_chart(fig)

        st.subheader("Métricas de Avaliação do Modelo")
        mae = np.mean(np.abs(forecast_mean - closing_prices[-passos_previsao:])) 
        rmse = np.sqrt(np.mean((forecast_mean - closing_prices[-passos_previsao:]) ** 2))  
        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")


