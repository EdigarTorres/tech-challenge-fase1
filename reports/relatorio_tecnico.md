# Relatório Técnico - Tech Challenge (Fase 1)
**Sistema Inteligente de Suporte ao Diagnóstico Oncológico (Câncer de Mama)**

---

## 1. Introdução e Definição do Problema Clínico
Hospitais universitários e centros de diagnóstico enfrentam um volume crescente de exames citopatológicos e de imagem. Neste projeto, abordamos a triagem automatizada de **Câncer de Mama** a partir de dados morfométricos extraídos de imagens digitalizadas de Punção Aspirativa por Agulha Fina (PAAF), utilizando a base pública **Breast Cancer Wisconsin (Diagnostic)**.

- **Tamanho da Base:** 569 pacientes e 30 variáveis numéricas contínuas (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal dos núcleos celulares, calculados na média, erro padrão e pior valor/`worst`).
- **Codificação Clínica da Variável Alvo (`target`):**
  - `1 = Maligno` (212 casos — 37,3%): Presença da doença maligna (foco prioritário da detecção).
  - `0 = Benigno` (357 casos — 62,7%): Ausência de malignidade.

---

## 2. Estratégias de Exploração (EDA) e Pré-processamento
1. **Qualidade dos Dados:** A inspeção confirmou ausência de valores nulos explícitos, mas revelou forte assimetria e presença de *outliers* naturais em variáveis de área (`mean area`, `worst area`) e concavidade.
2. **Análise de Correlação:** A matriz de correlação demonstrou alta colinearidade entre `radius`, `perimeter` e `area` (correlação > 0.98) e forte associação positiva entre `worst concave points`, `worst perimeter`, `worst radius` e o diagnóstico maligno (`r > 0.77`).
3. **Separação Estrita de Dados (Prevenção de *Data Leakage*):**
   - **Treino (70% - 398 amostras):** Ajuste dos transformadores e treinamento dos algoritmos.
   - **Validação (15% - 85 amostras):** Comparação preliminar e seleção de hiperparâmetros.
   - **Teste (15% - 86 amostras):** Avaliação final cega de generalização.
4. **Pipeline em Python (`scikit-learn`):**
   - `SimpleImputer(strategy='median')`: Garante resiliência em produção caso novos prontuários cheguem com campos laboratoriais não preenchidos, usando a mediana por ser robusta a *outliers*.
   - `StandardScaler()`: Padroniza todas as variáveis para média 0 e variância 1, requisito indispensável para a convergência e comparabilidade dos coeficientes da Regressão Logística.

---

## 3. Modelos Utilizados e Justificativa
Foram implementadas três técnicas distintas com ponderação de classes (`class_weight='balanced'`) para compensar o desbalanceamento (37,3% Malignos vs 62,7% Benignos):
1. **Regressão Logística:** Modelo linear clássico na área médica, excelente baseline e altamente estável quando as fronteiras de decisão após padronização são bem comportadas.
2. **Árvore de Decisão (`max_depth=4`):** Permite visualizar regras de corte diretas, porém apresenta maior variância isoladamente.
3. **Random Forest (`n_estimators=200, max_depth=6`):** Comitê não-linear de árvores de decisão que reduz a variância, captura interações complexas entre biomarcadores e possui suporte nativo e exato ao `TreeExplainer` da biblioteca **SHAP**.

---

## 4. Avaliação Crítica das Métricas (Foco no Recall)
Em triagem oncológica, os custos dos erros são assimétricos:
- **Falso Negativo (FN):** Paciente com tumor **Maligno** liberado como **Benigno**. É o erro mais perigoso, pois atrasa o tratamento e reduz a sobrevida.
- **Falso Positivo (FP):** Paciente **Benigno** encaminhado para revisão prioritária ou biópsia confirmatória. Gera custo operacional moderado, mas protege a vida.

Por isso, a escolha do modelo baseia-se primordialmente no **Recall (Sensibilidade)** aliado ao **F1-Score** e **ROC-AUC**.

---

## 5. Interpretabilidade (Feature Importance e SHAP)
Para eliminar o efeito "caixa-preta":
- **Análise Global (`SHAP Summary Plot`):** Demonstra que valores elevados de `worst radius`, `worst concave points`, `worst perimeter`, `worst area` e `mean concave points` aumentam drasticamente a probabilidade de malignidade.
- **Análise Local (`SHAP Waterfall Plot`):** Para cada paciente avaliado na interface Streamlit, o sistema gera uma explicação individual detalhando exatamente quanto cada variável contribuiu para afastar ou aproximar a predição do limiar de risco.

---

## 6. Aplicabilidade Prática no Hospital e Soberania Médica
O modelo **pode e deve ser utilizado na prática clínica** dentro das seguintes diretrizes éticas e operacionais:
1. **Sistema de Triagem e Priorização de Fila:** Assim que os dados citopatológicos são lançados no prontuário eletrônico, o modelo calcula a probabilidade de malignidade e classifica a urgência da revisão patológica.
2. **Apoio Visual à Decisão:** O médico recebe não apenas um número, mas os fatores de risco destacados via SHAP.
3. **Palavra Final do Médico:** O algoritmo **jamais emite laudo definitivo autônomo**. A decisão diagnóstica e terapêutica final permanece integralmente sob responsabilidade do médico especialista.
