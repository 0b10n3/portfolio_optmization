import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import base64


def carregar_svg_como_base64(caminho_arquivo):
    try:
        with open(caminho_arquivo, "rb") as f:
            conteudo_binario = f.read()
        conteudo_base64 = base64.b64encode(conteudo_binario).decode("utf-8")
        return f"data:image/svg+xml;base64,{conteudo_base64}"
    except FileNotFoundError:
        return "📈"
    except Exception as e:
        return "📈"


icone_pagina_svg = "📈"

st.set_page_config(
    page_title="Simulação de Portfólio",
    page_icon=icone_pagina_svg,
    layout="wide"
)

NM_PALETTE = {
    "forestDepthDark": "#102620",
    "primaryForestDepth": "#1A3C34",
    "charcoalSophistication": "#333333",
    "mintSerenityLight": "#EAF0E2",
    "mintSerenityDark": "#B8C8A8",
    "forestDepthLight": "#346C5C",
    "skyTrust": "#007BFF",
    "skyTrustLight": "#4DA3FF",
    "mintSerenity": "#D9E6C8",
    "oneOverNColor": "#FFD700",
    "pristineClarity": "#FFFFFF"
}

st.markdown(f"""
<style>
    /* Estilização geral do corpo */
    body {{
        color: {NM_PALETTE['mintSerenityLight']}; 
        background-color: {NM_PALETTE['forestDepthDark']}; 
        font-family: 'Inter', sans-serif;
    }}

    /* Contêiner principal do Streamlit (área de conteúdo) */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        border-radius: 15px;
        background-color: {NM_PALETTE['primaryForestDepth']}; 
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15); 
    }}

    /* Estilização da barra lateral */
    [data-testid="stSidebar"] {{
        background-color: {NM_PALETTE['forestDepthDark']}; 
        border-right: 1px solid {NM_PALETTE['forestDepthLight']}; 
    }}
    [data-testid="stSidebar"] .stTextInput > label,
    [data-testid="stSidebar"] .stDateInput > label,
    [data-testid="stSidebar"] .stNumberInput > label {{
        color: {NM_PALETTE['mintSerenity']} !important; 
        font-weight: bold;
    }}
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stDateInput > div > div > input,
    [data-testid="stSidebar"] .stNumberInput > div > div > input {{
        background-color: {NM_PALETTE['charcoalSophistication']}; 
        color: {NM_PALETTE['mintSerenityLight']}; 
        border: 1px solid {NM_PALETTE['forestDepthLight']}; 
        border-radius: 8px;
    }}

    /* Botões */
    .stButton > button {{
        background-color: {NM_PALETTE['mintSerenity']}; 
        color: {NM_PALETTE['forestDepthDark']}; 
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: {NM_PALETTE['mintSerenityLight']}; 
        color: {NM_PALETTE['forestDepthDark']};
    }}
    .stButton > button:active {{
        background-color: {NM_PALETTE['mintSerenityDark']}; 
        color: {NM_PALETTE['forestDepthDark']};
    }}

    /* Expansor */
    .streamlit-expanderHeader {{
        font-size: 1.1rem;
        color: {NM_PALETTE['mintSerenity']}; 
        font-weight: bold;
    }}
    .streamlit-expanderContent {{
        background-color: {NM_PALETTE['charcoalSophistication']}; 
        border-radius: 0 0 8px 8px;
        border: 1px solid {NM_PALETTE['forestDepthLight']}; 
        border-top: none;
        padding: 1rem;
        color: {NM_PALETTE['mintSerenityLight']}; 
    }}

    /* Dataframes */
    .stDataFrame {{ 
        border-radius: 8px;
        overflow: hidden;
    }}
    .stDataFrame table {{
        color: {NM_PALETTE['mintSerenityLight']}; 
    }}
    .stDataFrame thead th {{
        background-color: {NM_PALETTE['forestDepthLight']}; 
        color: {NM_PALETTE['mintSerenity']}; 
    }}
    .stDataFrame tbody tr:nth-child(even) {{
        background-color: {NM_PALETTE['charcoalSophistication']}; 
    }}
    .stDataFrame tbody tr:nth-child(odd) {{
        background-color: {NM_PALETTE['primaryForestDepth']}; 
    }}

    /* Fundo do gráfico Plotly */
    .plotly-graph-div {{
         border-radius: 10px;
         overflow: hidden;
    }}

    /* Títulos e Cabeçalhos */
    h1, h2, h3 {{
        color: {NM_PALETTE['mintSerenity']}; 
        font-weight: bold;
    }}
    h1 {{
        border-bottom: 2px solid {NM_PALETTE['forestDepthLight']}; 
        padding-bottom: 0.3em;
    }}
    h4, h5, h6 {{
        color: {NM_PALETTE['mintSerenityDark']}; 
    }}

    /* Mensagens específicas do Streamlit */
    .stAlert {{
        border-radius: 8px;
        color: {NM_PALETTE['forestDepthDark']}; 
    }}
    div[data-testid="stInfo"], div[data-testid="stSuccess"] {{
        background-color: {NM_PALETTE['mintSerenity']}; 
        border: 1px solid {NM_PALETTE['forestDepthLight']}; 
        color: {NM_PALETTE['primaryForestDepth']}; 
    }}
    div[data-testid="stInfo"] svg, div[data-testid="stSuccess"] svg {{ fill: {NM_PALETTE['primaryForestDepth']}; }}
    div[data-testid="stWarning"] {{
        background-color: #FFF3E0; 
        border: 1px solid #FFA000;
        color: #C66900;
    }}
    div[data-testid="stWarning"] svg {{ fill: #FFA000; }}
    div[data-testid="stError"] {{
        background-color: #FFEBEE; 
        border: 1px solid #D32F2F;
        color: #B71C1C;
    }}
    div[data-testid="stError"] svg {{ fill: #D32F2F; }}

    /* Rótulos e valores de métricas */
    .stMetric > label {{
        color: {NM_PALETTE['mintSerenityDark']} !important; 
    }}
    .stMetric > div[data-testid="stMetricValue"] {{
        color: {NM_PALETTE['mintSerenityLight']} !important; 
    }}

</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_stock_data(tickers, start_date, end_date):
    """
    Search for stock data using yfinance API.
    """

    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        all_data = yf.download(tickers, start=start_dt, end=end_dt, progress=False, auto_adjust=False)

        if all_data.empty:
            st.error(f"Nenhum dado para os tickers: {tickers}. Verifique se os tickers são válidos.")
            return pd.DataFrame()

        price_data = pd.DataFrame()
        if isinstance(all_data.columns, pd.MultiIndex):
            if 'Adj Close' in all_data.columns.levels[0]:
                price_data = all_data['Adj Close']
            elif 'Close' in all_data.columns.levels[0]:
                price_data = all_data['Close']
                st.sidebar.info(
                    "Usando preços de 'Fechamento' (Close) pois 'Fechamento Ajustado' (Adj Close) não disponível.")
            else:
                st.error(
                    "Não foram encontrados dados de 'Fechamento Ajustado' (Adj Close) nem de 'Fechamento' (Close).")
                return pd.DataFrame()
        elif isinstance(all_data, pd.DataFrame):  # Caso de um único ticker
            if 'Adj Close' in all_data.columns:
                price_data = all_data[['Adj Close']]
            elif 'Close' in all_data.columns:
                price_data = all_data[['Close']]
                if len(tickers) == 1: st.sidebar.info(
                    f"Usando preço de 'Fechamento' (Close) para {tickers[0]} pois 'Fechamento Ajustado' (Adj Close) não estava disponível.")
            else:
                st.error(
                    f"Não foram encontrados dados de 'Fechamento Ajustado' (Adj Close) nem de 'Fechamento' (Close) para o ticker: {tickers[0] if tickers else 'Desconhecido'}")
                return pd.DataFrame()

            # Renomear coluna para o nome do ticker se for apenas um
            if len(tickers) == 1 and not price_data.empty:
                price_data.columns = tickers
        else:  # Caso inesperado
            st.error(f"Os dados baixados possuem um formato inesperado. Tipo: {type(all_data)}")
            return pd.DataFrame()

        price_data = price_data.dropna()
        if price_data.empty:
            st.warning(
                f"Nenhum dado de preço não-NaN encontrado após limpeza para os tickers: {tickers} no período selecionado.")
            return pd.DataFrame()
        return price_data
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        return pd.DataFrame()


def calculate_returns(data):
    if data.empty: return pd.DataFrame()
    return data.pct_change().dropna()


def run_monte_carlo_simulation(returns, num_portfolios, risk_free_rate):
    if returns.empty or len(returns.columns) == 0:
        st.error("Não é possível executar a simulação com dados de retorno vazios ou inválidos.")
        return pd.DataFrame(), None, None, None

    num_assets = len(returns.columns)
    asset_names = [str(col) for col in returns.columns]

    results_mc = np.zeros((3 + num_assets, num_portfolios))
    mean_returns_annualized = returns.mean() * 252
    cov_matrix_annualized = returns.cov() * 252

    if cov_matrix_annualized.isnull().values.any() or (num_assets > 1 and np.linalg.det(cov_matrix_annualized) == 0):
        st.warning("A matriz de covariância contém NaN ou é singular. A simulação pode não ser concluída.")

    progress_bar_sidebar = st.sidebar.progress(0)
    status_text_sidebar = st.sidebar.empty()
    status_text_sidebar.text("Iniciando Simulação de Monte Carlo...")

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns_annualized.values * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annualized, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev if portfolio_stddev != 0 else (
            np.inf if portfolio_return > risk_free_rate else (-np.inf if portfolio_return < risk_free_rate else 0))

        results_mc[0, i] = portfolio_return
        results_mc[1, i] = portfolio_stddev
        results_mc[2, i] = sharpe_ratio
        for j in range(num_assets): results_mc[j + 3, i] = weights[j]

        if (i + 1) % (num_portfolios // 100 or 1) == 0:
            progress_percentage = int(((i + 1) / num_portfolios) * 100)
            progress_bar_sidebar.progress(progress_percentage)
            status_text_sidebar.text(f"Executando Simulação: {progress_percentage}% Completo")

    status_text_sidebar.text("Simulação de Monte Carlo Completa!")
    time.sleep(1)
    progress_bar_sidebar.empty()
    status_text_sidebar.empty()

    column_names_mc = ['Retorno', 'Volatilidade', 'Sharpe Ratio'] + asset_names
    results_df = pd.DataFrame(results_mc.T, columns=column_names_mc)

    # Calcular Portfólio 1/N
    one_over_n_weights_array = np.array([1 / num_assets] * num_assets)
    one_over_n_return = np.sum(mean_returns_annualized.values * one_over_n_weights_array)
    one_over_n_volatility = np.sqrt(
        np.dot(one_over_n_weights_array.T, np.dot(cov_matrix_annualized, one_over_n_weights_array)))
    one_over_n_sharpe = (
                                    one_over_n_return - risk_free_rate) / one_over_n_volatility if one_over_n_volatility != 0 else (
        np.inf if one_over_n_return > risk_free_rate else (-np.inf if one_over_n_return < risk_free_rate else 0))

    one_over_n_portfolio_data = {
        'Retorno': one_over_n_return,
        'Volatilidade': one_over_n_volatility,
        'Sharpe Ratio': one_over_n_sharpe,
        'Pesos': pd.Series(one_over_n_weights_array, index=asset_names)  # Pesos como pd.Series
    }

    return results_df, mean_returns_annualized, cov_matrix_annualized, one_over_n_portfolio_data


def find_optimal_portfolios(results_df):
    if results_df.empty: return None, None
    results_df['Sharpe Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    max_sharpe_portfolio = None
    if not results_df['Sharpe Ratio'].isnull().all():
        max_sharpe_portfolio = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    min_vol_portfolio = None
    if not results_df['Volatilidade'].isnull().all():
        min_vol_portfolio = results_df.loc[results_df['Volatilidade'].idxmin()]
    return max_sharpe_portfolio, min_vol_portfolio


def plot_efficient_frontier(results_df, max_sharpe_portfolio, min_vol_portfolio, one_over_n_portfolio, palette):
    if results_df.empty:
        st.warning("Sem resultados da simulação para plotar.")
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=results_df['Volatilidade'], y=results_df['Retorno'], mode='markers',
        marker=dict(
            color=results_df['Sharpe Ratio'],
            colorscale='Plasma',
            showscale=True,
            size=7,
            opacity=0.7,
            colorbar=dict(
                title=dict(text='Índice de Sharpe', font=dict(color=palette['mintSerenityLight'])),
                tickfont=dict(color=palette['mintSerenityLight'])
            )
        ),
        name='Portfólios Simulados',
        customdata=results_df.drop(columns=['Retorno', 'Volatilidade', 'Sharpe Ratio']),
        hovertemplate='<b>Volatilidade</b>: %{x:.3f}<br><b>Retorno</b>: %{y:.3f}<br><b>Índice de Sharpe</b>: %{marker.color:.3f}<br><extra></extra>'
    ))
    if max_sharpe_portfolio is not None and not max_sharpe_portfolio.empty:
        fig.add_trace(
            go.Scattergl(x=[max_sharpe_portfolio['Volatilidade']], y=[max_sharpe_portfolio['Retorno']], mode='markers',
                         marker=dict(color=palette['skyTrustLight'], size=12, symbol='star',
                                     line=dict(width=1, color=palette['pristineClarity'])),
                         name='Máx. Índice de Sharpe'))
    if min_vol_portfolio is not None and not min_vol_portfolio.empty:
        fig.add_trace(
            go.Scattergl(x=[min_vol_portfolio['Volatilidade']], y=[min_vol_portfolio['Retorno']], mode='markers',
                         marker=dict(color=palette['mintSerenity'], size=12, symbol='diamond',
                                     line=dict(width=1, color=palette['forestDepthDark'])),
                         name='Mín. Volatilidade'))
    if one_over_n_portfolio is not None:
        fig.add_trace(
            go.Scattergl(x=[one_over_n_portfolio['Volatilidade']], y=[one_over_n_portfolio['Retorno']], mode='markers',
                         marker=dict(color=palette['oneOverNColor'], size=10, symbol='triangle-up',
                                     line=dict(width=1, color=palette['forestDepthDark'])),
                         name='Portfólio 1/N'))

    fig.update_layout(
        title=dict(text='Simulação da Fronteira Eficiente', font=dict(color=palette['mintSerenity'])),
        xaxis_title=dict(text='Volatilidade Anualizada (Desvio Padrão)', font=dict(color=palette['mintSerenityDark'])),
        yaxis_title=dict(text='Retorno Anualizado', font=dict(color=palette['mintSerenityDark'])),
        xaxis=dict(tickformat='.1%', gridcolor=palette['forestDepthLight'],
                   tickfont=dict(color=palette['mintSerenityDark'])),
        yaxis=dict(tickformat='.1%', gridcolor=palette['forestDepthLight'],
                   tickfont=dict(color=palette['mintSerenityDark'])),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=palette['mintSerenityLight']),
            bgcolor=NM_PALETTE['primaryForestDepth'],
            bordercolor=palette['forestDepthLight'],
            borderwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Fundo da área de plotagem transparente
        paper_bgcolor='rgba(0,0,0,0)', # Fundo geral do gráfico transparente
        font=dict(color=palette['mintSerenityLight'])
    )
    return fig


with st.sidebar:
    st.header("Parâmetros do Portfólio")
    ticker_string = st.text_input(
        "Tickers das Ações (separados por vírgula)",
        "PETR4.SA,VALE3.SA,WEGE3.SA,B3SA3.SA"
    )
    tickers = [ticker.strip().upper() for ticker in ticker_string.split(',') if ticker.strip()]

    col_sd, col_ed = st.columns(2)
    with col_sd:
        start_date_input = st.date_input("Data de Início",
                                         pd.to_datetime("2023-01-01").date())
    with col_ed:
        end_date_input = st.date_input("Data Final", pd.Timestamp.now().normalize().date())

    num_portfolios = st.number_input("Número de Simulações", min_value=100, max_value=10000, value=2000,
                                     step=100)
    risk_free_rate = st.number_input("Taxa Livre de Risco (Anualizada)", min_value=0.0, max_value=0.5, value=0.10,
                                     step=0.005, format="%.4f")

    run_button = st.button("Executar Simulação")

st.title("Simulação de Portfólios - Fronteira Eficiente")
st.markdown("""
Este painel interativo é uma ferramenta de apoio aos seus estudos sobre a 
Teoria Moderna do Portfólio, desenvolvida por Harry Markowitz. Com base 
nos seus conhecimentos sobre a relação entre risco e retorno, e a 
importância da diversificação, você poderá simular a construção de diversas
 carteiras de ativos. Ao inserir os códigos dos ativos desejados, o período
 histórico para análise e outros parâmetros como a taxa livre de risco e o
 número de simulações de Monte Carlo, a ferramenta calculará e visualizará 
 a Fronteira Eficiente. Este gráfico representa o conjunto de portfólios 
 ótimos que oferecem o maior retorno esperado para um determinado nível de 
 risco (volatilidade), permitindo identificar carteiras que maximizam o 
 Índice de Sharpe ou que apresentam a menor volatilidade possível. 
 Utilize os controles na barra lateral para configurar sua simulação e 
 explore como diferentes combinações de ativos impactam o perfil de 
 risco-retorno do seu portfólio.
""")

if run_button:
    if not tickers:
        st.warning("Por favor, insira pelo menos um ticker de ação na barra lateral.")
    elif start_date_input >= end_date_input:
        st.warning("A data final deve ser posterior à data de início.")
    else:
        st.info(f"Buscando dados para: {', '.join(tickers)} de {start_date_input} até {end_date_input}")
        stock_data = get_stock_data(tickers, start_date_input, end_date_input)

        if not stock_data.empty:
            st.success("Dados buscados com sucesso!")
            returns = calculate_returns(stock_data)

            if returns.empty or len(returns) < 2:
                st.warning(
                    "Não há dados históricos válidos suficientes para calcular os retornos (são necessários pelo menos 2 dias).")
            else:
                with st.spinner("Executando Simulação de Monte Carlo... Isso pode levar um momento."):
                    results_df, mean_returns, cov_matrix, one_over_n_portfolio = run_monte_carlo_simulation(returns,
                                                                                                            num_portfolios,
                                                                                                            risk_free_rate)

                if not results_df.empty:
                    st.success("Simulação completa!")
                    max_sharpe_portfolio, min_vol_portfolio = find_optimal_portfolios(results_df)

                    st.header("Resultados")
                    st.subheader("Portfólios Simulados")
                    fig_ef = plot_efficient_frontier(results_df, max_sharpe_portfolio, min_vol_portfolio,
                                                     one_over_n_portfolio, NM_PALETTE)
                    st.plotly_chart(fig_ef, use_container_width=True)

                    st.subheader("Portfólios de Referência")
                    col_msp, col_mvp, col_1_n = st.columns(3)

                    if max_sharpe_portfolio is not None and not max_sharpe_portfolio.empty:
                        with col_msp:
                            st.markdown("#### Máximo Índice de Sharpe")
                            st.metric("Retorno", f"{max_sharpe_portfolio['Retorno']:.2%}")
                            st.metric("Volatilidade", f"{max_sharpe_portfolio['Volatilidade']:.2%}")
                            st.metric("Índice de Sharpe", f"{max_sharpe_portfolio['Sharpe Ratio']:.2f}")
                            st.markdown("##### Pesos:")
                            weights_msp = max_sharpe_portfolio.drop(['Retorno', 'Volatilidade', 'Sharpe Ratio'])
                            st.dataframe(weights_msp.map('{:.2%}'.format), use_container_width=True)
                    else:
                        with col_msp:
                            st.warning("Não foi possível determinar o portfólio de Máximo Índice de Sharpe.")

                    if min_vol_portfolio is not None and not min_vol_portfolio.empty:
                        with col_mvp:
                            st.markdown("#### Mínima Volatilidade")
                            st.metric("Retorno", f"{min_vol_portfolio['Retorno']:.2%}")
                            st.metric("Volatilidade", f"{min_vol_portfolio['Volatilidade']:.2%}")
                            st.metric("Índice de Sharpe", f"{min_vol_portfolio['Sharpe Ratio']:.2f}")
                            st.markdown("##### Pesos:")
                            weights_mvp = min_vol_portfolio.drop(['Retorno', 'Volatilidade', 'Sharpe Ratio'])
                            st.dataframe(weights_mvp.map('{:.2%}'.format), use_container_width=True)
                    else:
                        with col_mvp:
                            st.warning("Não foi possível determinar o portfólio de Mínima Volatilidade.")

                    if one_over_n_portfolio is not None:
                        with col_1_n:
                            st.markdown("#### Portfólio 1/N (Igual Ponderação)")
                            st.metric("Retorno", f"{one_over_n_portfolio['Retorno']:.2%}")
                            st.metric("Volatilidade", f"{one_over_n_portfolio['Volatilidade']:.2%}")
                            st.metric("Índice de Sharpe", f"{one_over_n_portfolio['Sharpe Ratio']:.2f}")
                            st.markdown("##### Pesos:")
                            st.dataframe(one_over_n_portfolio['Pesos'].map('{:.2%}'.format), use_container_width=True)

                    with st.expander("Ver Preços Selecionados Brutos (Fech. Ajust. ou Fech.)"):
                        st.dataframe(stock_data, use_container_width=True)
                    with st.expander("Ver Dados de Retornos Diários"):
                        st.dataframe(returns, use_container_width=True)
                    with st.expander(f"Ver os 10 Melhores Resultados da Simulação (de {num_portfolios})"):
                        st.dataframe(results_df.head(10), use_container_width=True)
                else:
                    st.error("A simulação falhou em produzir resultados.")
        else:
            st.error("A busca de dados falhou ou não retornou dados de preço utilizáveis.")
else:
    st.info("Configure seu portfólio na barra lateral e clique em 'Executar Simulação de Otimização'.")

st.markdown("---")
st.markdown("Simulação de Portfólios")