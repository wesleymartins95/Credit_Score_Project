# 📝 Notas do Projeto

## 🎯 Objetivo
Desenvolver modelo de credit scoring para a Quod com foco em:
- Predição de inadimplência
- Identificação de fatores de risco
- Insights acionáveis

## 📅 Timeline
- Início: [09/02/2026]
- Recesso Carnaval: [14/02/2026 a 18/02/2026]
- Prazo: [06/03/2026]
- Status: Semana 3/5

## ✅ Decisões Tomadas

### Tratamento de Dados
1. **Missing Values:**
   - Colunas com >50% missing: removidas.

   - Numéricas: imputação pela mediana, mas em variáveis críticas de renda (renda_mensal, comprometimento_renda) 
foram aplicadas uma imputação segmentada por faixa etária (mediana dentro de cada grupo etário) + fallback para mediana geral. 
Além disso, foi criado flags de missing para preservar a informação da ausência.

   - Categóricas: A coluna faixa_etaria, não foi imputada pela moda. Em vez disso, foi criado uma flag (faixa_etaria_missing) e imputado como "não informado", ou seja, foi adicionado uma categoria explícita para os casos faltantes.
Isso é até mais robusto do que usar moda, porque evita mascarar o missing e permite que o modelo aprenda que a ausência é um sinal em si.

2. **Outliers:**
   - Identificados mas mantidos inicialmente
   - Reavaliação após primeira modelagem

3. **Balanceamento:**
   - Usar SMOTE para treino
   - Manter dados originais para validação

### Features Selecionadas
[atrasos_30dias - atrasos_90dias - dependentes]

## 📊 Métricas de Sucesso
- AUC-ROC > 0.75
- Recall para inadimplentes > 70%
- Precision > 65%

## ❓ Dúvidas/Pendências
- [ ] Definir threshold de aprovação
- [ ] Validar estratégia de cross-validation
- [ ] Confirmar features finais

## 📚 Referências
- Kaggle Competition: Give Me Some Credit
- Documentação sklearn: https://...
- Papers sobre credit scoring: [links]




