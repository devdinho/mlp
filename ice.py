import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class TitanicPredictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Predição de Sobrevivência do Titanic")
        self.window.geometry("700x800")
        
        # Valores padrão para os campos
        self.default_values = {
            'pclass': '1',
            'sex': 'Masculino',
            'age': '30',
            'sibsp': '0',
            'parch': '0'
        }
        
        # Carregando e preparando os dados
        self.load_and_prepare_data()
        
        # Criando e treinando o modelo
        self.create_and_train_model()
        
        # Calculando MAE
        self.calculate_mae()
        
        # Criando a interface
        self.create_interface()
        
    def load_and_prepare_data(self):
        # Carregando os dados
        df = pd.read_csv('titanic.csv')
        
        # Codificando variáveis categóricas e excluindo as colunas "Fare" e "Embarked"
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
        # Separando recursos e rótulos, sem "Fare" e "Embarked"
        X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
        y = df['Survived']
        
        # Tratando valores ausentes (opcional)
        X['Age'].fillna(X['Age'].mean(), inplace=True)
        
        # Dividindo dados em conjuntos de treinamento e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.7, random_state=42
        )
        
        # Padronizando os dados
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def create_and_train_model(self):
        # Criando o modelo MLP com Scikit-Learn
        self.model = MLPClassifier(hidden_layer_sizes=(8, 8), activation='relu', max_iter=1000, random_state=42)
        
        # Treinando o modelo
        self.model.fit(self.X_train, self.y_train)

    def calculate_mae(self):
        # Fazendo predições no conjunto de teste
        y_pred = self.model.predict(self.X_test)
        # Calculando MAE
        self.mae = mean_absolute_error(self.y_test, y_pred)
        
    def create_interface(self):
        # Estilo
        style = ttk.Style()
        style.configure('TLabel', padding=5, font=('Helvetica', 10))
        style.configure('TButton', padding=5, font=('Helvetica', 10))
        style.configure('TEntry', padding=5)
        
        # Frame principal
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = ttk.Label(main_frame, text="Predição de Sobrevivência do Titanic", 
                              font=('Helvetica', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Mostrando MAE
        mae_label = ttk.Label(main_frame, 
                            text=f"Erro Médio Absoluto do Modelo: {self.mae:.4f}",
                            font=('Helvetica', 10))
        mae_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Campos de entrada
        # Classe
        ttk.Label(main_frame, text="Classe:").grid(row=2, column=0, sticky=tk.W)
        self.pclass = ttk.Combobox(main_frame, values=[1, 2, 3], width=30)
        self.pclass.grid(row=2, column=1, pady=5)
        self.pclass.set(self.default_values['pclass'])
        
        # Sexo
        ttk.Label(main_frame, text="Sexo:").grid(row=3, column=0, sticky=tk.W)
        self.sex = ttk.Combobox(main_frame, values=['Masculino', 'Feminino'], width=30)
        self.sex.grid(row=3, column=1, pady=5)
        self.sex.set(self.default_values['sex'])
        
        # Idade
        ttk.Label(main_frame, text="Idade:").grid(row=4, column=0, sticky=tk.W)
        self.age = ttk.Entry(main_frame, width=33)
        self.age.grid(row=4, column=1, pady=5)
        self.age.insert(0, self.default_values['age'])
        
        # SibSp
        ttk.Label(main_frame, text="Número de Irmãos/Cônjuges:").grid(row=5, column=0, sticky=tk.W)
        self.sibsp = ttk.Entry(main_frame, width=33)
        self.sibsp.grid(row=5, column=1, pady=5)
        self.sibsp.insert(0, self.default_values['sibsp'])
        
        # Parch
        ttk.Label(main_frame, text="Número de Pais/Filhos:").grid(row=6, column=0, sticky=tk.W)
        self.parch = ttk.Entry(main_frame, width=33)
        self.parch.grid(row=6, column=1, pady=5)
        self.parch.insert(0, self.default_values['parch'])
        
        # Frame para botões
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        # Botões
        predict_button = ttk.Button(button_frame, text="Fazer Predição", 
                                  command=self.make_prediction)
        predict_button.grid(row=0, column=0, padx=10)
        
        reset_button = ttk.Button(button_frame, text="Limpar Campos", 
                                command=self.reset_fields)
        reset_button.grid(row=0, column=1, padx=10)
        
        # Frame para resultados
        self.result_frame = ttk.LabelFrame(main_frame, text="Resultados da Predição", padding="10")
        self.result_frame.grid(row=8, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Labels para resultados
        self.result_label = ttk.Label(self.result_frame, text="", font=('Helvetica', 12, 'bold'))
        self.result_label.grid(row=0, column=0, pady=10)

    def reset_fields(self):
        # Limpa todos os campos para os valores padrão
        self.pclass.set(self.default_values['pclass'])
        self.sex.set(self.default_values['sex'])
        self.age.delete(0, tk.END)
        self.age.insert(0, self.default_values['age'])
        self.sibsp.delete(0, tk.END)
        self.sibsp.insert(0, self.default_values['sibsp'])
        self.parch.delete(0, tk.END)
        self.parch.insert(0, self.default_values['parch'])
        
        # Limpa os resultados
        self.result_label.config(text="")
        
    def make_prediction(self):
        try:
            # Coletando dados do formulário
            input_data = [
                int(self.pclass.get()),
                1 if self.sex.get() == 'Feminino' else 0,
                float(self.age.get()),
                int(self.sibsp.get()),
                int(self.parch.get())
            ]
            
            # Padronizando os dados de entrada
            input_scaled = self.scaler.transform([input_data])
            
            # Fazendo a predição e obtendo a probabilidade de sobrevivência
            probability = self.model.predict_proba(input_scaled)[0][1]
            survival_percentage = probability * 100
            
            # Atualizando o resultado com a porcentagem de sobrevivência
            result_text = (f"Predição Final: {survival_percentage:.2f}% de chance de sobrevivência.\n"
                           f"O passageiro provavelmente {'sobreviveu!' if probability > 0.5 else 'não sobreviveu.'}")
            
            self.result_label.config(
                text=result_text,
                foreground='green' if probability > 0.5 else 'red'
            )
            
        except ValueError as e:
            messagebox.showerror("Erro", "Por favor, verifique se todos os campos foram preenchidos corretamente.")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TitanicPredictor()
    app.run()