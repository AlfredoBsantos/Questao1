import joblib
from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado
try:
    model = joblib.load('loan_approval_model.pkl')
    print("Modelo 'loan_approval_model.pkl' carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None # Define model como None para evitar erros se o carregamento falhar

@app.route('/')
def home():
    # Renderiza o template HTML do formulário
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Erro: Modelo não carregado. Contate o administrador.")

    # Obter os dados do formulário
    gender = request.form['gender']
    married = request.form['married']
    dependents = request.form['dependents']
    education = request.form['education']
    self_employed = request.form['self_employed']
    applicant_income = int(request.form['applicant_income'])
    loan_amount = int(request.form['loan_amount'])

    # Criar um DataFrame com os dados de entrada
    # É CRÍTICO que as colunas e a ordem sejam as mesmas que o modelo esperava durante o treinamento
    # As colunas após o pré-processamento (remoção) eram:
    # 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount'

    # Note que 'Dependents' foi tratado como categórico durante o OHE, então ele precisa ser uma string aqui
    # Os valores de entrada devem corresponder aos tipos de dados esperados pelo pré-processador
    data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents], # Mantenha como string para OneHotEncoder
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'LoanAmount': [loan_amount]
    }
    input_df = pd.DataFrame(data)

    # Realizar a predição
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Mapear a predição para 'Sim' ou 'Não'
    # Lembre-se que LabelEncoder mapeou 'N' para 0 e 'Y' para 1.
    loan_status = "Aprovado" if prediction == 1 else "Negado"

    # Probabilidade da classe prevista
    # A probabilidade de `prediction_proba[0]` é para a classe 0 (Negado)
    # A probabilidade de `prediction_proba[1]` é para a classe 1 (Aprovado)
    predicted_class_proba = prediction_proba[prediction] * 100 # Multiplica por 100 para porcentagem

    result_text = f"Status do Empréstimo: {loan_status} (Probabilidade: {predicted_class_proba:.2f}%)"

    return render_template('index.html', prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True) # debug=True para desenvolvimento, reinicia o servidor ao salvar