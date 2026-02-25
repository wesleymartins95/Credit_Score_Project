# 📅 Plano Detalhado - Semana 3: Feature Engineering

## 🎯 Objetivo da Semana
Criar features derivadas que aumentem o poder preditivo do modelo

## 📋 Tarefas por Dia

### Segunda/Terça (Dia 23/24)
**Feature Engineering - Parte 1**
- [ ] Razões e proporções (ex: dívida/renda)
- [ ] Variáveis de idade/tempo
- [ ] Agregações (soma, média de múltiplas features)
- **Entregável:** 10+ novas features criadas

### Quarta/Quinta (Dia 25/26)
**Feature Engineering - Parte 2**
- [ ] Binning de variáveis contínuas
- [ ] Interações entre features
- [ ] Flags booleanas
- **Entregável:** Dataset completo com features

### Sexta/Sabado (Dia 27/28 )
**Encoding e Transformações**
- [ ] One-hot encoding de categóricas
- [ ] Scaling/normalização
- [ ] Transformações (log, sqrt)
- **Entregável:** Dados prontos para modelagem

### Domingo/Segunda (Dia 01/03 e 02/03)
**Seleção de Features**
- [ ] Remover features correlacionadas
- [ ] Feature importance (tree-based)
- [ ] Recursive Feature Elimination
- **Entregável:** Lista final de features

### Terca/Quarta (Dia 03/03 e 04/03)
**Preparação Final**
- [ ] Train/test split
- [ ] Aplicar SMOTE
- [ ] Salvar datasets finais
- **Entregável:** Dados prontos para semana 4

## 📊 Métricas de Sucesso
- Criar mínimo 15 features novas
- Reduzir features de XX para ~30 mais relevantes
- Dataset balanceado pronto

## 🔧 Ferramentas
- sklearn.preprocessing
- feature-engine
- imblearn (SMOTE)