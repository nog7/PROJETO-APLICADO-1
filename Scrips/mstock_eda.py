# %%
# Importação de bibliotecas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings

# Configurações iniciais
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Análise Exploratória de Dados",
    page_icon=":bar_chart:", 
    layout="wide",
)

# Carregar dados
df = pd.read_csv('C:\\Users\\euller.nogueira\\Documents\\Atividades\\PROJETO APLICADO I\\Notebooks\\Dataset\\Microsoft_Stock.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

def load_css(file_name):
    with open(file_name) as f:
        return f.read()

st.markdown(f'<style>{load_css("style.css")}</style>', unsafe_allow_html=True)

# Sidebar para navegação
st.sidebar.markdown('<h1 style="text-align: center;">Navegação</h1>', unsafe_allow_html=True)
if 'page' not in st.session_state:
    st.session_state.page = 'Informações dos Dados'

# Navegação via botões na barra lateral
if st.sidebar.button('Informações dos Dados', key='info'):
    st.session_state.page = 'Informações dos Dados'
if st.sidebar.button('Análise Exploratória dos Dados', key='analysis'):
    st.session_state.page = 'Análise Exploratória dos Dados'
if st.sidebar.button('Previsão com SARIMAX', key='sarimax'):
    st.session_state.page = 'Previsão com SARIMAX'

# Páginas com conteúdo dinâmico
if st.session_state.page == 'Informações dos Dados':
    st.header('Informações dos Dados')

    st.subheader('Quantidade de Linhas e Colunas:')
    num_rows, num_cols = df.shape
    st.write(f'O dataset possui **{num_rows} linhas** e **{num_cols} colunas**.')

    st.subheader('Visão Inicial dos Dados (Head):')
    st.dataframe(df.head())
    st.caption('''O Head do dataset apresenta as primeiras cinco linhas dos dados, permitindo uma visão inicial das variáveis e seus valores. Esta visualização 
               é útil para entender a estrutura do dataset, identificar os tipos de dados presentes em cada coluna e garantir que os dados foram carregados corretamente.''')

    st.subheader('Visão Final dos Dados (Tail):')
    st.dataframe(df.tail())
    st.caption('''O Tail do dataset exibe as últimas cinco linhas dos dados, proporcionando uma visão das entradas finais. Esta análise ajuda a verificar se os dados terminam corretamente, sem valores ausentes ou erros no carregamento, 
               e pode revelar padrões ou tendências nas entradas mais recentes, especialmente em séries temporais, como no caso de preços de ações e volumes de negociação.''')

    st.subheader('Estatísticas Descritivas:')
    st.write(df.describe())
    st.caption('''A tabela de estatísticas descritivas fornece informações importantes sobre as colunas numéricas do dataset, incluindo medidas como média, 
               desvio padrão, valores mínimo e máximo, além de percentis. Esses indicadores ajudam a entender a distribuição e a variação dos dados.''')

    st.subheader('Verificação de Valores Ausentes:')
    st.write(df.isnull().sum())
    st.caption('''Esta seção destaca o número de valores ausentes em cada coluna do dataset. 
               A análise de valores nulos é essencial para identificar problemas de qualidade dos dados que podem afetar os resultados das análises.''')

elif st.session_state.page == 'Análise Exploratória dos Dados':
    st.header('Análise Exploratória dos Dados')

    # Seleção de tipo de análise
    analysis_option = st.selectbox(
        'Escolha a análise a ser exibida:',
        ['Gráficos de Séries Temporais', 'Preço Mais Alto e Mais Baixo', 
         'Preço de Fechamento e Média Móvel', 'Mudança Diária no Preço', 
         'Volume de Negociação']
    )

    # Gráficos de Séries Temporais
    if analysis_option == 'Gráficos de Séries Temporais':
        st.subheader('Gráficos de Séries Temporais')
        columns = ['Open', 'Close', 'High', 'Low']
        colors = ['blue', 'purple', 'green', 'red']
        fig = make_subplots(rows=2, cols=2, subplot_titles=columns)

        for i, (col, color) in enumerate(zip(columns, colors)):
            row = i // 2 + 1
            col_num = i % 2 + 1
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode='lines', name=col,
                line=dict(color=color)
            ), row=row, col=col_num)

        fig.update_layout(title='Séries Temporais das Ações da Microsoft', height=800, width=1000)
        st.plotly_chart(fig)
        st.caption('''Os gráficos de séries temporais apresentam a evolução dos preços das ações ao longo do tempo para as variáveis *Open*, *Close*, *High* e *Low*. 
                   Esses gráficos permitem identificar padrões e tendências, como ciclos de alta e baixa, volatilidade e momentos de estabilidade no mercado. 
                   O uso de subplots facilita a visualização individual de cada variável, permitindo comparações diretas entre elas.''')

    # Preço Mais Alto e Mais Baixo
    elif analysis_option == 'Preço Mais Alto e Mais Baixo':
        st.subheader('Preço Mais Alto e Mais Baixo')

        def plot_combined_series(df, columns, titles, colors):
            fig = go.Figure()
            for column, title, color in zip(columns, titles, colors):
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[column], mode='lines', name=title,
                    line=dict(color=color, width=1.5)
                ))
            fig.update_layout(
                title='Série Temporal: Maior e Menor Preço',
                xaxis_title='Data', yaxis_title='Valores',
                legend_title='Preços', height=600, width=900
            )
            return fig

        combined_fig = plot_combined_series(df, ['High', 'Low'], 
                                            ['Preço Mais Alto', 'Preço Mais Baixo'], 
                                            ['green', 'red'])
        st.plotly_chart(combined_fig)
        st.caption('''Este gráfico combina os preços máximos (*High*) e mínimos (*Low*) em uma única série temporal. 
                   Ele mostra a amplitude de variação dos preços das ações ao longo do tempo, permitindo identificar períodos de alta volatilidade ou estabilidade. 
                   O eixo X representa as datas, enquanto o eixo Y indica os valores correspondentes em dólares americanos. 
                   Essa visualização é útil para entender a dinâmica dos preços extremos e sua relação com eventos específicos.''')

    # Preço de Fechamento e Média Móvel
    elif analysis_option == 'Preço de Fechamento e Média Móvel':
        st.subheader('Preço de Fechamento e Média Móvel')
        df['Moving Average'] = df['Close'].rolling(window=180).mean()

        def plot_moving_average(df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'], mode='lines', name='Preço de Fechamento',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Moving Average'], mode='lines',
                name='Média Móvel Semestral', line=dict(color='orange')
            ))
            fig.update_layout(
                title='Preço de Fechamento e Média Móvel',
                xaxis_title='Data', yaxis_title='Preço (USD)',
                legend_title='Legenda', height=600, width=900
            )
            return fig

        moving_avg_fig = plot_moving_average(df)
        st.plotly_chart(moving_avg_fig)
        st.caption('''O gráfico apresenta a evolução do preço de fechamento (*Close*) das ações junto com a média móvel semestral (*Moving Average*). 
                   A média móvel é uma técnica de suavização que reduz a influência de flutuações curtas e destaca tendências de longo prazo. 
                   A combinação das duas linhas permite identificar momentos em que o preço está acima ou abaixo da média, indicando possíveis zonas de compra ou venda. 
                   ''')

    # Mudança Diária no Preço
    elif analysis_option == 'Mudança Diária no Preço':
        st.subheader('Mudança Diária no Preço')
        df['Mudança Diária'] = df['Close'] - df['Open']

        def plot_daily_change(df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Mudança Diária'], mode='lines',
                name='Mudança Diária', line=dict(color='orange')
            ))
            fig.update_layout(
                title='Mudança Diária no Preço',
                xaxis_title='Data', yaxis_title='Mudança (USD)',
                legend_title='Legenda', height=600, width=900
            )
            return fig

        daily_change_fig = plot_daily_change(df)
        st.plotly_chart(daily_change_fig)
        st.caption('''O gráfico de mudança diária no preço apresenta a diferença entre os preços de fechamento (*Close*) e abertura (*Open*) para cada dia. 
                   Valores positivos indicam dias de alta, enquanto valores negativos representam dias de queda. 
                   Esse gráfico é útil para avaliar a volatilidade diária do mercado e identificar padrões sazonais ou anomalias que podem ser investigadas mais profundamente.''')

    # Volume de Negociação
    elif analysis_option == 'Volume de Negociação':
        st.subheader('Volume de Negociação')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Volume'], mode='lines', name='Volume de Negociação',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='Volume de Negociação ao Longo do Tempo',
            xaxis_title='Data', yaxis_title='Volume',
            legend_title='Legenda', height=600, width=900
        )
        st.plotly_chart(fig)
        st.caption('''O gráfico de Volume de Negociações apresenta a quantidade de ações negociadas diariamente ao longo do tempo. 
                   Ele é uma métrica fundamental para avaliar o interesse e a atividade do mercado em relação a um ativo específico, 
                   neste caso, as ações da Microsoft.''')

elif st.session_state.page == 'Previsão com SARIMAX':
    st.header('Previsão com SARIMAX')

    closing_prices = df['Close']
    p, d, q = 1, 1, 1  
    P, D, Q, s = 1, 1, 1, 12 
    model = SARIMAX(closing_prices, order=(p, d, q), seasonal_order=(P, D, Q, s))
    sarimax_model = model.fit(disp=False)

    forecast_steps = 365
    forecast = sarimax_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=closing_prices, mode='lines', name='Preço de Fechamento',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_mean, mode='lines', name='Previsão (SARIMAX)',
        line=dict(color='orange')
    ))
    fig.update_layout(
        title='Previsão de Preço de Fechamento com SARIMAX',
        xaxis_title='Data', yaxis_title='Preço (USD)',
        legend_title='Legenda', height=600, width=900
    )
    st.plotly_chart(fig)
    st.caption('''O SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) é um modelo usado para prever séries temporais, 
               considerando efeitos sazonais e variáveis externas. Ele é composto por três componentes principais: 
               AR (AutoRegressivo), I (Integrado) e MA (Média Móvel). 
               O AR usa dados passados para prever o futuro, o I aplica diferenciação para tornar a série estacionária, 
               removendo tendências ao calcular as diferenças entre valores consecutivos. 
               O MA ajusta as previsões com base nos erros passados (resíduos) para corrigir desvios. 
               A parte exógena (X) permite incorporar fatores externos que afetam a série, 
               tornando o SARIMAX ideal para séries com padrões sazonais e influências externas.''')
