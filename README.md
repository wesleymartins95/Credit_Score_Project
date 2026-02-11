

# 🎯 Projeto de Credit Scoring e Análise de Risco de Crédito

> Projeto de Machine Learning para predição de inadimplência e análise de risco de crédito

---

## 📋 Sobre o Projeto

Este projeto desenvolve um **modelo de credit scoring** utilizando técnicas de Machine Learning para prever inadimplência 
e apoiar decisões de concessão de crédito.

### 🎯 Objetivos

- ✅ Prever inadimplência com acurácia superior a 80%
- ✅ Identificar principais fatores de risco
- ✅ Criar scorecard interpretável para decisões de crédito
- ✅ Gerar insights acionáveis para o negócio

### 🏢 Contexto de Negócio

O modelo visa reduzir perdas por inadimplência enquanto mantém uma taxa saudável de aprovação de crédito, equilibrando risco e oportunidade de receita.

---

## 📊 Dataset

**Fonte:** [Kaggle - Give Me Some Credit](link)

**Características:**
- **Registros:** 150.000 clientes
- **Features:** 13 variáveis
- **Target:** SeriousDlqin2yrs (inadimplência nos últimos 2 anos)

---

## 🗂️ Estrutura do Projeto
```
credit-scoring-analysis/
│
├── data/
│   ├── raw/              # Dados originais
│   ├── processed/        # Dados processados
│   └── features/         # Features engineered
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # ✅ EDA
│   ├── 02_data_preprocessing.ipynb    # 🔄 Em andamento
│   ├── 03_feature_engineering.ipynb   # 📅 Próximo
│   └── 04_modeling.ipynb              # 📅 Planejado
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── modeling.py
│
├── models/              # Modelos treinados
├── reports/             # Relatórios e apresentações
├── dashboards/          # Dashboard interativo
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.13.11**
- **Pandas & NumPy** - Manipulação de dados
- **Scikit-learn** - Modelagem ML
- **XGBoost/LightGBM** - Modelos avançados
- **Matplotlib/Seaborn** - Visualizações
- **SHAP** - Interpretabilidade
- **Streamlit** - Dashboard interativo

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/credit-scoring-analysis.git
cd credit-scoring-analysis
```

### 2. Crie ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

No Anaconda Navigator você não precisa usar venv. O fluxo é:
Abra o Anaconda Navigator.
Vá até a aba Environments.
Clique em Create.
Escolha:
Nome do ambiente (ex: Credit_Score_Project)

Versão do Python desejada (ex: 3.13.11)

(Opcional) pacotes iniciais a instalar

Clique em Create e pronto: o ambiente aparece na lista.
```

### 3. Instale dependências
```bash
pip install -r requirements.txt
```

### 4. Execute os notebooks
```bash
jupyter notebook
```

---

## 📈 Progresso do Projeto

- [x] Setup do ambiente
- [x] Análise exploratória (EDA)
- [ ] Pré-processamento de dados
- [ ] Feature engineering
- [ ] Modelagem
- [ ] Avaliação e otimização
- [ ] Dashboard interativo
- [ ] Documentação final

---

## 💡 Principais Insights (Até Agora)

1. **Desbalanceamento de classes:** 6,7% inadimplentes vs XX% bons pagadores
2. **Variáveis mais correlacionadas:** - [atrasos_30dias - atrasos_90dias - dependentes]
3. **Dados faltantes:** [comprometimento_renda	29749	19.83% - renda_mensal	29731	19.82% - faixa_etaria	688	0.46% - utilizacao_credito	114	0.08% - divida_ratio	18	0.01% ] colunas requerem tratamento


---

## 📧 Contato

**Wesley Martins**  
📧 wesleyat@outlook.com  
💼 [LinkedIn](https://www.linkedin.com/in/wesleymartinsdados/)  
🐙 [GitHub](https://github.com/wesleymartins95)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

**Desenvolvido com 💙 por [Wesley Martins]**