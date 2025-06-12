# Projeto de Classificação de Aprovação de Empréstimos (Questão 1 - Aprendizagem de Máquina)

## Visão Geral do Projeto

Este repositório contém a solução para a Questão 1 da avaliação da disciplina de Aprendizagem de Máquina da Universidade Federal do Rio Grande do Norte (UFRN) - Escola Agrícola de Jundiaí. [cite_start]O objetivo principal é desenvolver um modelo de Machine Learning para a classificação binária de aprovação de empréstimos, com base no dataset `loan.csv`, e implantar este modelo em uma aplicação web desenvolvida com Flask.

## Detalhamento das Etapas Cumpridas (Questão 1)

Todas as etapas sugeridas na Questão 1 foram rigorosamente cumpridas, conforme detalhado abaixo:

1.  **Carregamento e Balanceamento das Classes (Item a):**
    * [cite_start]O dataset de empréstimos (`loan.csv`) foi carregado a partir da URL fornecida no GitHub.
    * A variável alvo ('loan_status') teve seu balanceamento verificado. [cite_start]Dada a natureza desbalanceada dos dados, a técnica `SMOTE` (Synthetic Minority Over-sampling Technique) foi aplicada dentro dos pipelines de treinamento para balancear as classes da variável alvo.

2.  **Remoção de Colunas (Item b):**
    * [cite_start]As colunas `Loan_ID`, `CoapplicantIncome`, `Loan_Amount_Term`, `Credit_History` e `Property_Area` foram removidas do dataset, conforme especificado.

3.  **Tratamento de Dados Faltantes (Itens c e d):**
    * [cite_start]Os dados faltantes nas colunas `Dependents`, `Self_Employed`, `Married` e `Gender` foram preenchidos com o valor majoritário (moda) de suas respectivas colunas, como sugerido.
    * A coluna `LoanAmount`, que também possuía dados faltantes, foi preenchida com a mediana.
    * Após estas operações, foi confirmada a ausência de dados faltantes em todo o dataset processado.

4.  **Transformação de Features Categóricas e Variável Target (Item e):**
    * Features categóricas como `Gender`, `Married`, `Education` e `Self_Employed` foram transformadas usando `OneHotEncoder` dentro do `ColumnTransformer` para convertê-las em um formato numérico.
    * [cite_start]A variável alvo `Loan_Status` foi transformada utilizando `LabelEncoder` (mapeando 'N' para 0 e 'Y' para 1).

5.  **Normalização da Feature 'Dependents' (Item f):**
    * Para a feature `Dependents`, o rótulo '3+' foi associado ao número '3', conforme a instrução. [cite_start]Os demais números (0, 1, 2) foram mantidos.

6.  **Padronização de Features Numéricas (Item g):**
    * As features numéricas restantes (`ApplicantIncome`, `LoanAmount`) foram padronizadas utilizando `StandardScaler`, garantindo que tivessem média zero e desvio padrão um, o que é crucial para muitos algoritmos de Machine Learning.

7.  **Criação de Pipelines, Grid Search e Validação Cruzada (Item h):**
    * Foram criados 5 pipelines distintos de transformação e treinamento, cada um incorporando o pré-processamento (`ColumnTransformer` e `SMOTE`) e um dos seguintes algoritmos de classificação: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting e SVC.
    * Para cada pipeline, foi utilizado `GridSearchCV` com validação cruzada (`StratifiedKFold` com `n_splits=5`) para otimizar os hiperparâmetros.
    * As métricas de desempenho comparadas foram o `classification_report` e a `AUC` da curva ROC, avaliando a capacidade de generalização dos modelos.

8.  **Escolha e Salvamento do Modelo Final (Item i):**
    * O modelo **Decision Tree** foi escolhido como o modelo final. Apesar de apresentar sinais de overfitting (alto desempenho no treino vs. baixo na validação), sua capacidade de aprendizado e a estrutura transparente o tornaram a escolha para demonstrar a pipeline completa.
    * O pipeline completo do modelo Decision Tree (o `best_estimator_` resultante do `GridSearchCV`) foi salvo utilizando a biblioteca `joblib` no arquivo `loan_approval_model.pkl`. Este arquivo contém todas as etapas de pré-processamento e o modelo treinado, pronto para ser consumido na aplicação.

9.  **Criação da Aplicação Web em Flask (Item j e k):**
    * Foi criada uma aplicação web completa (frontend e backend) utilizando o framework Flask em Python.
    * [cite_start]**Frontend (`templates/index.html`):** Desenvolvido com um formulário HTML contendo todos os controles solicitados: `Sexo` (radiobutton), `Dependentes` (option/selection), `Casado` (radio), `Autônomo` (radio), `Rendimento do Requerente` (text), `Educação` (radio/selection) e `Valor do Empréstimo` (text).
    * **Backend (`app.py`):** Responsável por carregar o modelo `loan_approval_model.pkl`, receber os dados do formulário, passá-los pelo pipeline de pré-processamento e fazer a predição.
    * [cite_start]A aplicação exibe o status da análise de empréstimo ("Aprovado" ou "Negado") e a probabilidade associada à classe majoritária classificada, conforme requerido.

## Estrutura do Repositório

```Questao1/
├── app.py                      # Backend da aplicação Flask
├── loan_approval_model.pkl     # Modelo de Machine Learning serializado (Decision Tree)
├── Questao1.ipynb              # Notebook Jupyter com o pipeline de ML (treinamento e avaliação)
├── .gitignore                  # Define arquivos e pastas a serem ignorados pelo Git
└── templates/
└── index.html              # Frontend HTML da aplicação web
```
## Como Executar a Aplicação

Para executar a aplicação Flask localmente, siga os passos abaixo:

### Pré-requisitos

Certifique-se de ter o [Anaconda](https://www.anaconda.com/products/individual) instalado, que inclui o `conda` para gerenciamento de ambientes e pacotes.

### Configuração do Ambiente Python

1.  **Abra o Anaconda Prompt** (ou seu terminal no Linux/macOS).
2.  **Crie e ative o ambiente virtual** com as dependências necessárias. Este ambiente garante que todas as bibliotecas tenham versões compatíveis:

    ```bash
    conda create -n loan_ml_env python=3.10 scikit-learn=1.2.2 imbalanced-learn=0.10.1 pandas numpy Flask joblib matplotlib seaborn jupyterlab -c conda-forge -y
    conda activate loan_ml_env
    ```
    (Aguarde a conclusão da criação e instalação dos pacotes. Pode levar alguns minutos.)

3.  **Confirme as versões (Opcional):**
    ```bash
    pip show scikit-learn
    pip show imbalanced-learn
    pip show Flask
    ```

### Execução do Projeto

1.  **Navegue até o diretório do projeto:**
    ```bash
    cd C:\Users\alfre\OneDrive\Documentos\Questao1
    ```
    (Se você moveu a pasta, use o caminho correto para onde ela está localizada.)

2.  **Abra e execute o Notebook Jupyter (Uma única vez):**
    * Para garantir que o `loan_approval_model.pkl` esteja salvo com a versão correta do `scikit-learn` (1.2.2) do ambiente `loan_ml_env`, inicie o JupyterLab a partir deste ambiente:
        ```bash
        jupyter lab
        ```
    * No JupyterLab, abra o `Questao1.ipynb`.
    * Vá em `Kernel` -> `Restart Kernel and Run All Cells`. Isso irá re-executar todo o treinamento e salvar o modelo `loan_approval_model.pkl` novamente.
    * Após a execução completa, feche o JupyterLab (pressione `Ctrl + C` no terminal onde ele foi iniciado para encerrar o servidor).

3.  **Execute a Aplicação Flask:**
    * Com o ambiente `loan_ml_env` ainda ativo e no diretório do projeto, execute o arquivo `app.py`:
        ```bash
        python app.py
        ```
    * O Flask iniciará um servidor local. Você verá uma mensagem no terminal como: `* Running on http://127.0.0.1:5000`.

4.  **Acesse a Aplicação no Navegador:**
    * Abra seu navegador web e acesse `http://127.0.0.1:5000`.
    * Preencha o formulário com os dados do requerente do empréstimo e clique em "Prever Aprovação".

## Contato

Para dúvidas ou sugestões, por favor, entre em contato.



