

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
│   ├── 02_data_preprocessing.ipynb    # ✅ Em andamento
│   ├── 03_feature_engineering.ipynb   # ✅ Próximo
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

## ✅ Checklist de Progresso

### Semana 1-2: Preparação e EDA
- [x] Setup do projeto
- [x] Análise exploratória inicial
- [x] Análise bivariada
- [x] Information Value
- [x] Limpeza de dados
- [x] Documentação

### Semana 3: Feature Engineering (Próxima)
- [x] **Encoding:** transformação de variáveis categóricas em formato numérico
- [x] **Scaling:** geração de 3 versões do dataset (`no_scale`, `standardized`, `normalized`)
- [x] **Seleção de Features:** redução para ~40 variáveis mais relevantes após análise
- [x] **Split:** divisão estratificada em treino/teste (80/20)
- [x] **Balanceamento:** aplicação de SMOTE apenas no treino
      
#### 📊 Datasets Finais
data/final/
├── no_scale_train_balanced.csv       # Para Random Forest, XGBoost
├── standardized_train_balanced.csv   # Para Logistic Regression
├── normalized_train_balanced.csv     # Para Neural Networks
├── [versão]_test.csv                 # Sempre usar para avaliação
└── [versão]_train_original.csv       # Opcional para comparação

### Semana 4: Modelagem
- [ ] Baseline model
- [ ] Modelos avançados
- [ ] Otimização
- [ ] Avaliação

### Semana 5: Entrega
- [ ] Dashboard
- [ ] Apresentação
- [ ] Documentação final

---

## 💡 Principais Insights (Até Agora)

1. **Desbalanceamento de classes:** 6,7% inadimplentes vs 93,3% bons pagadores.- **Ação:** Aplicar SMOTE na modelagem
2. **Variáveis mais correlacionadas com Inadimplência:** - [atrasos_30dias - atrasos_90dias - dependentes]
3. **Dados faltantes:** [comprometimento_renda	29749	19.83% - renda_mensal	29731	19.82% - faixa_etaria	688	0.46% - utilizacao_credito	114	0.08% - divida_ratio	18	0.01% ] colunas requerem tratamento
4. **Variáveis com Maior Poder Preditivo (IV > 0.3)**
- `[Utilização de crédito]`: IV = 1,12 - [Continua sendo o principal discriminador de risco, clientes que usam crédito de forma intensa têm padrões distintos de inadimplência. Decisão: deve ser o pilar central do score.]
- `[Dependentes]`: IV = 0,48 - [Forte poder discriminatório, maior número de dependentes pressiona orçamento familiar e aumenta risco.
Decisão: incluir como variável-chave, possivelmente ajustada por renda.]
- `[Atrasos em 30 dias]`: IV = 0,47 - [Forte preditor de inadimplência, histórico de atraso é altamente indicativo de risco futuro.
Decisão: manter como variável crítica no score.]

#### 2. Tratamento de Dados
- **Colunas removidas por excesso de missing (>50%):**
- `Renda mensal e Comprometimento da renda apresentavam cerca de 20% de valores faltantes, mas não chegaram a 50%.Portanto, nenhuma coluna foi removida por excesso de missing.`
`(Insight: o dataset manteve todas as variáveis originais, mas com imputação e flags para monitorar qualidade.)`

- **Valores imputados usando mediana:**
- `Renda mensal → imputada pela mediana segmentada por faixa etária.`
- `Comprometimento da renda → imputada pela mediana segmentada.`
- `Faixa etária → imputada em casos raros (~0,4%).`
`(Insight: imputação recuperou variáveis antes inutilizáveis, como renda mensal, que ganhou relevância pós-limpeza – IV subiu de 0,08 → 0,12.)`

- **Outliers identificados (mantidos para avaliação):**
- `Idade → valores fora de [18–90] truncados e sinalizados com idade_outlier_flag.`
- `Renda mensal → valores acima de R$ 50.000 truncados e sinalizados com renda_outlier_flag.`
- `Comprometimento da renda → valores acima de 100% truncados e sinalizados com comprometimento_outlier_flag.`
`(Insight: flags não discriminam risco isoladamente (IV ≈ 0), mas são úteis para governança e podem ser exploradas em interações.)`

🔑 Principais Insights da Semana 3
- `Feature Engineering: criação de mais de 60 novas variáveis aumentou a capacidade do modelo de capturar relações complexas entre renda, dívidas e histórico de crédito.`

- `Encoding & Scaling: diferentes versões do dataset permitem testar modelos variados (Logistic Regression, Random Forest, Neural Networks) sem perda de consistência.`

- `Seleção de Features: redução para ~40 variáveis mais relevantes garante modelos mais enxutos, interpretáveis e com menor risco de overfitting.`

- `Balanceamento com SMOTE: corrigiu a desproporção entre classes, aumentando a representatividade de clientes inadimplentes no treino e melhorando a robustez dos modelos.`

- `Preparação Final: datasets prontos e documentados, com versões balanceadas e originais, asseguram comparações justas na etapa de modelagem.`


---
✅ Semana 4: Modelagem - COMPLETA!
Modelos Desenvolvidos
✅ Logistic Regression (Baseline) – AUC: 0.7195

✅ Random Forest – AUC: 0.8012

✅ XGBoost – AUC: 0.8427

✅ LightGBM – AUC: 0.8572

✅ Modelo Final Otimizado (LightGBM Tuned) – AUC: 0.8572, KS: 0.5609

Melhoria Alcançada
Baseline: 0.7195

Final: 0.8572

Ganho Relativo: +19.13%

Próxima Semana
Semana 5: Dashboard e Apresentação Final

Construção de dashboard interativo (Streamlit).

Preparação da apresentação executiva para a Quod.

Consolidação do portfólio no GitHub.

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