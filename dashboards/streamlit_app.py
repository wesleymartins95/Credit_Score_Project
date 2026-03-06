"""
Credit Scoring Dashboard
Aplicação Streamlit para análise e predição de crédito
Desenvolvido para Quod DataTech
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score
)

# Função KS
def ks_statistic(y_true, y_pred_proba):
    data = pd.DataFrame({"y": y_true, "p": y_pred_proba}).sort_values("p")
    cum_pos = np.cumsum(data["y"] == 1) / (data["y"] == 1).sum()
    cum_neg = np.cumsum(data["y"] == 0) / (data["y"] == 0).sum()
    return np.max(np.abs(cum_pos - cum_neg))

# Configuração da página
st.set_page_config(page_title="Credit Scoring Dashboard", page_icon="💳", layout="wide")

# Sidebar
st.sidebar.title("🎯 Navegação")
page = st.sidebar.radio(
    "Selecione a página:",
    ["🏠 Visão Geral", "📊 Performance do Modelo", "🔮 Fazer Predição", 
     "📈 Análise de Features", "💰 Impacto de Negócio"]
)

# Carregar dados e modelo
@st.cache_data
def load_data():
    test_df = pd.read_csv("data/final/no_scale_test.csv")
    results_df = pd.read_csv("reports/model_results.csv")
    return test_df, results_df



@st.cache_resource
def load_model():
    import requests, joblib
    from io import BytesIO

    # IDs dos arquivos no Google Drive
    files = {
        "xgboost.pkl": "1YY2rI1lpuyBZ9IJ5yymENjcP26J89dqA",
        "lightgbm.pkl": "1UJEWaVwzJlw4h2g-v1mVvE1Cj05vVjZQ",
        "log_reg.pkl": "1lCnFQDa7uD05n9j73NQN-c3X9D3_s5d4",
        "log_reg_pipeline.pkl": "1XOuB8YbzbCiIILmZTGGGED0JXPFEn501",
        "df_woe.pkl": "1N4zteAuLjqxzfc3dcpyLuEcZkJUUGqlW",
        "woe_comprometimento_renda.pkl": "1yYE9k7ejuumKOcCJi9wUdArvBv6QNoAt"
    }

    models = {}
    for name, fid in files.items():
        url = f"https://drive.google.com/uc?id={fid}"
        response = requests.get(url)
        response.raise_for_status()
        models[name] = joblib.load(BytesIO(response.content))

    # Retorna todos os modelos em um dicionário
    return models

# Uso no app
test_df, results_df = load_data()
models = load_model()


# Variáveis principais
target_col = 'inadipl_90dias_ult2anos'
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ks = ks_statistic(y_test, y_pred_proba)

baseline_auc = results_df[results_df['model_name'].str.contains('Baseline')]['test_auc'].values[0]

# ============================================================================
# PÁGINA 1: VISÃO GERAL
# ============================================================================
if page == "🏠 Visão Geral":
    st.title("💳 Credit Scoring Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("AUC-ROC", f"{auc:.4f}", f"+{((auc - baseline_auc)/baseline_auc)*100:.1f}% vs Baseline")
    with col2: st.metric("KS-Statistic", f"{ks:.4f}", "Benchmark ≥ 0.30")
    with col3: st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col4: st.metric("F1-Score", f"{f1:.4f}")
    with col5: st.metric("Total Predições", f"{len(y_test):,}")

    st.subheader("📈 Evolução dos Modelos")
    results_sorted = results_df.sort_values('test_auc')
    fig = go.Figure(go.Bar(
        x=results_sorted['test_auc'], y=results_sorted['model_name'],
        orientation='h', text=results_sorted['test_auc'].round(4), textposition='auto',
        marker=dict(color=results_sorted['test_auc'], colorscale='Viridis')
    ))
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# PÁGINA 2: PERFORMANCE DO MODELO
# ============================================================================
elif page == "📊 Performance do Modelo":
    st.title("📊 Performance do Modelo")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'Modelo (AUC={auc:.4f}, KS={ks:.4f})',
                             line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             name='Random', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig, width='stretch')

    from sklearn.metrics import precision_recall_curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    fig = go.Figure(go.Scatter(x=recall_vals, y=precision_vals, mode='lines',
                               line=dict(color='green', width=3)))
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# PÁGINA 3: FAZER PREDIÇÃO
# ============================================================================
elif page == "🔮 Fazer Predição":
    st.title("🔮 Simulador de Crédito")
    with st.form("prediction_form"):
        idade = st.number_input("Idade", min_value=18, max_value=100, value=35)
        renda = st.number_input("Renda Mensal (R$)", min_value=0.0, value=5000.0, step=100.0)
        divida_renda = st.slider("Debt Ratio", 0.0, 2.0, 0.3, 0.01)
        linhas_credito = st.number_input("Linhas de Crédito Abertas", 0, 50, 5)
        atrasos = st.number_input("Total de Atrasos", 0, 20, 0)
        submitted = st.form_submit_button("🎯 Fazer Predição")

    if submitted:

        input_df = pd.DataFrame({
        "idade": [idade],
        "renda_mensal": [renda],
        "divida_ratio": [divida_renda],
        "linhas_credito_abertas": [linhas_credito],
        "indice_severidade_atrasos": [atrasos]
    })

    # =============================
    # FEATURE ENGINEERING
    # =============================

    input_df["utilizacao_credito"] = input_df["divida_ratio"]

    input_df["alta_utilizacao_flag"] = (input_df["utilizacao_credito"] > 0.5).astype(int)

    input_df["renda_per_capita"] = input_df["renda_mensal"]

    input_df["baixa_renda_por_pessoa"] = (input_df["renda_mensal"] < 2000).astype(int)

    input_df["utilizacao_media_linha"] = input_df["utilizacao_credito"] / (input_df["linhas_credito_abertas"] + 1)

    input_df["renda_disponivel"] = input_df["renda_mensal"] * (1 - input_df["divida_ratio"])

    input_df["comprometimento_renda_ajustado"] = input_df["divida_ratio"] * input_df["renda_mensal"]

    input_df["utilizacao_credito_bin"] = (input_df["utilizacao_credito"] > 0.7).astype(int)

    input_df["interacao_idade_renda"] = input_df["idade"] * input_df["renda_mensal"]

    input_df["interacao_divida_linhas"] = input_df["divida_ratio"] * input_df["linhas_credito_abertas"]

    input_df["pressao_dependentes_renda"] = 0

    input_df["idade_quadrado"] = input_df["idade"] ** 2

    input_df["log_renda_mensal"] = np.log1p(input_df["renda_mensal"])

    input_df["utilizacao_credito_quadrado"] = input_df["utilizacao_credito"] ** 2

    input_df["score_interno"] = 500

    input_df["flag_possui_imovel"] = 0

    input_df["flag_thin_file"] = (input_df["linhas_credito_abertas"] < 3).astype(int)

    input_df["renda_mensal_missing"] = 0

    input_df["estabilidade_financeira"] = input_df["renda_mensal"] / (input_df["linhas_credito_abertas"] + 1)

    input_df["flag_cliente_estavel"] = (input_df["indice_severidade_atrasos"] == 0).astype(int)

    input_df["flag_alto_risco"] = (input_df["indice_severidade_atrasos"] > 5).astype(int)

    input_df["comprometimento_outlier_flag"] = (input_df["divida_ratio"] > 1).astype(int)

    input_df["emprestimos_imobiliarioss"] = 0

    # =============================
    # ORDEM DAS FEATURES DO MODELO
    # =============================

    expected_features = [
        'utilizacao_credito',
        'indice_severidade_atrasos',
        'alta_utilizacao_flag',
        'renda_per_capita',
        'baixa_renda_por_pessoa',
        'idade',
        'utilizacao_media_linha',
        'renda_disponivel',
        'comprometimento_renda_ajustado',
        'renda_mensal',
        'linhas_credito_abertas',
        'divida_ratio',
        'utilizacao_credito_bin',
        'interacao_idade_renda',
        'interacao_divida_linhas',
        'pressao_dependentes_renda',
        'idade_quadrado',
        'log_renda_mensal',
        'utilizacao_credito_quadrado',
        'score_interno',
        'flag_possui_imovel',
        'flag_thin_file',
        'renda_mensal_missing',
        'estabilidade_financeira',
        'flag_cliente_estavel',
        'flag_alto_risco',
        'comprometimento_outlier_flag',
        'emprestimos_imobiliarioss'
    ]

    input_df = input_df[expected_features]

    # =============================
    # PREDIÇÃO
    # =============================

    prob = model.predict_proba(input_df)[:, 1][0]

    st.subheader("📊 Resultado da Análise")

    st.metric("Probabilidade de Inadimplência", f"{prob:.2%}")

    if prob < 0.3:
        st.success("✅ Baixo risco de inadimplência")
    elif prob < 0.6:
        st.warning("⚠️ Risco moderado")
    else:
        st.error("🚨 Alto risco de inadimplência")


# ============================================================================
# PÁGINA 4: ANÁLISE DE FEATURES
# ============================================================================
elif page == "📈 Análise de Features":
    st.title("📈 Análise de Features")
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = feature_importance.head(15)
    fig = go.Figure(go.Bar(
        x=top_features['importance'], y=top_features['feature'],
        orientation='h', marker=dict(color=top_features['importance'], colorscale='Viridis'),
        text=top_features['importance'].round(4), textposition='auto'
    ))
    st.plotly_chart(fig, width='stretch')

    st.subheader("📊 Distribuição de uma Feature")
    selected_feature = st.selectbox("Selecione uma feature:", options=X_test.columns.tolist())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=test_df[test_df[target_col]==0][selected_feature], name="Bom Pagador", opacity=0.7))
    fig.add_trace(go.Histogram(x=test_df[test_df[target_col]==1][selected_feature], name="Inadimplente", opacity=0.7))
    fig.update_layout(barmode='overlay')
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# PÁGINA 5: IMPACTO DE NEGÓCIO
# ============================================================================
elif page == "💰 Impacto de Negócio":
    st.title("💰 Impacto de Negócio")
    st.subheader("⚙️ Configurações Financeiras")
    valor_medio_credito = st.number_input("Valor Médio de Crédito (R$)", min_value=1000.0, value=5000.0, step=500.0)
    taxa_perda = st.slider("Taxa de Perda (%)", min_value=0, max_value=100, value=70)/100
    custo_oportunidade = st.number_input("Custo de Oportunidade (R$)", min_value=0.0, value=500.0, step=100.0)

    threshold = st.slider("Threshold de decisão:", 0.0, 1.0, 0.5, 0.05)
    y_pred_business = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_business).ravel()

    # Cálculos financeiros
    perda_fn = fn * valor_medio_credito * taxa_perda
    perda_fp = fp * custo_oportunidade
    economia_tp = tp * valor_medio_credito * taxa_perda
    ganho_tn = tn * custo_oportunidade * 0.1
    impacto_total = economia_tp + ganho_tn - perda_fn - perda_fp

    st.subheader("💵 Impacto Financeiro Estimado")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Economia (TP)", f"R$ {economia_tp:,.2f}")
    with col2: st.metric("Perda (FN)", f"R$ {perda_fn:,.2f}")
    with col3: st.metric("Oportunidade Perdida (FP)", f"R$ {perda_fp:,.2f}")
    with col4: st.metric("Impacto Líquido", f"R$ {impacto_total:,.2f}")

    # Gráfico waterfall
    fig = go.Figure(go.Waterfall(
        name="Impacto",
        orientation="v",
        measure=["relative","relative","relative","relative","total"],
        x=["Economia (TP)", "Perda (FN)", "Oportunidade Perdida (FP)", "Ganho (TN)", "Total"],
        y=[economia_tp, -perda_fn, -perda_fp, ganho_tn, impacto_total],
        connector={"line":{"color":"rgb(63,63,63)"}}
    ))
    fig.update_layout(title="Análise de Impacto Financeiro", showlegend=False, height=400)
    st.plotly_chart(fig, width='stretch')

    # Projeção anual
    st.subheader("📊 Projeção Anual")
    volume_mensal = st.number_input("Volume Mensal de Análises", min_value=100, value=1000, step=100)
    custo_implementacao = st.number_input("Custo de Implementação (R$)", min_value=0.0, value=50000.0, step=5000.0)

    impacto_por_analise = impacto_total / len(y_test)
    impacto_mensal = impacto_por_analise * volume_mensal
    impacto_anual = impacto_mensal * 12
    roi_anual = ((impacto_anual - custo_implementacao) / custo_implementacao) * 100 if custo_implementacao > 0 else 0
    payback_meses = custo_implementacao / impacto_mensal if impacto_mensal > 0 else float('inf')

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Impacto Anual", f"R$ {impacto_anual:,.2f}")
    with col2: st.metric("ROI Anual", f"{roi_anual:.1f}%")
    with col3: st.metric("Payback", f"{payback_meses:.1f} meses" if payback_meses != float('inf') else "N/A")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>Credit Scoring Dashboard</strong></p>
    <p>Desenvolvido para Quod DataTech | Projeto de Machine Learning</p>
    <p>📧 seuemail@dominio.com | 💼 linkedin.com/in/seulinkedin | 🐙 github.com/seugithub</p>
</div>
""", unsafe_allow_html=True)
