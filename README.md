# 🤖 Employee Churn Prediction

This project aims to predict employee attrition using machine learning techniques applied to real-world Human Resources data.

---

## 🌐 Dataset source

**IBM HR Analytics Employee Attrition & Performance**  
🔗 [Available on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

The dataset contains information on 1,470 employees, including salary, tenure, job satisfaction, age, overtime, and performance.

---

## 🔢 Project steps

### 1️⃣ Load data  
Read and inspect the original dataset.

### 2️⃣ Exploratory Data Analysis (EDA)  
Analyze distributions, correlations, and the target variable (`Attrition`).

### 3️⃣ Cleaning & encoding  
Encode categorical variables and prepare the data for modeling.

### 4️⃣ Train/test split  
Split the dataset using stratified sampling to preserve class distribution.

### 5️⃣ Model training  
Train a Random Forest classification model.

### 6️⃣ Model evaluation  
Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

### 7️⃣ Feature importance  
Identify the most relevant features in predicting churn.

### 8️⃣ Conclusions and recommendations  
Generate actionable insights for business decision-making.

---

## 📊 Key results

- 🎯 **Model accuracy**: 82.99%
- 📉 The "Yes attrition" class had low sensitivity (recall: 0.11)
- 🔍 Key features: `OverTime`, `MonthlyIncome`, `TotalWorkingYears`, `JobLevel`, `YearsAtCompany`

---

## 💡 Technical recommendations

- Use `class_weight='balanced'` or SMOTE to handle class imbalance.
- Try other models such as XGBoost or Logistic Regression.
- Evaluate using ROC-AUC to assess class separability.

---

## 📌 Business insights

- The model helps detect risk patterns associated with employee turnover.
- Attrition is strongly linked to overtime, low salary, and short tenure.
- HR departments can use these insights to develop better retention strategies.

---

## 🛠️ Tools used

- Python
- Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- Jupyter Notebook

---

## 📁 Project structure

employee-churn-prediction/
├── data/
│ └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│ ├── 01_EDA.ipynb
│ └── 02_Modeling.ipynb
├── images/
├── requirements.txt
└── README.md


---

## 👩‍💻 Author

Project developed by **Nabila Isabel Padilla Resendiz**  
💼 Mechatronics Engineer | 📊 Aspiring Data Scientist | 💡 Passionate about learning and technology  

🔗 [LinkedIn](https://www.linkedin.com/in/nabilap/)  
💻 [GitHub](https://github.com/mecatronabi)

---

---

# 🤖 Predicción de Rotación de Empleados

Este proyecto tiene como objetivo predecir la rotación de empleados utilizando técnicas de Machine Learning aplicadas a datos reales del área de Recursos Humanos.

---

## 🌐 Fuente de datos

**IBM HR Analytics Employee Attrition & Performance**  
🔗 [Disponible en Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

Contiene información de 1,470 empleados, incluyendo salario, antigüedad, satisfacción laboral, edad, horas extra y desempeño.

---

## 🔢 Pasos del proyecto

### 1️⃣ Carga de datos  
Lectura y revisión del dataset original.

### 2️⃣ Exploración de datos (EDA)  
Análisis de distribuciones, correlaciones y la variable objetivo (`Attrition`).

### 3️⃣ Limpieza y codificación  
Codificación de variables categóricas y preparación de los datos.

### 4️⃣ División del dataset  
Separación en entrenamiento y prueba con muestreo estratificado.

### 5️⃣ Entrenamiento del modelo  
Entrenamiento de un modelo de clasificación con Random Forest.

### 6️⃣ Evaluación del modelo  
Evaluación usando accuracy, precision, recall, f1-score y matriz de confusión.

### 7️⃣ Importancia de variables  
Identificación de las variables más influyentes en la predicción.

### 8️⃣ Conclusiones y recomendaciones  
Generación de recomendaciones técnicas y de negocio.

---

## 📊 Resultados destacados

- 🎯 **Precisión del modelo**: 82.99%
- 📉 La clase "Sí renuncia" tuvo baja sensibilidad (recall: 0.11)
- 🔍 Variables clave: `OverTime`, `MonthlyIncome`, `TotalWorkingYears`, `JobLevel`, `YearsAtCompany`

---

## 💡 Recomendaciones técnicas

- Usar `class_weight='balanced'` o técnicas como SMOTE para abordar el desbalance.
- Probar otros algoritmos como XGBoost o regresión logística.
- Evaluar con ROC-AUC para analizar la separabilidad entre clases.

---

## 📌 Conclusiones de negocio

- El modelo permite identificar patrones de riesgo asociados a la rotación de personal.
- La rotación está relacionada con horas extra, bajo ingreso y poca antigüedad.
- Recursos Humanos puede usar estos insights para diseñar estrategias de retención efectivas.

---

## 🛠️ Tecnologías utilizadas

- Python
- Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- Jupyter Notebook

---

## 📁 Estructura del proyecto

employee-churn-prediction/
├── data/
│ └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│ ├── 01_EDA.ipynb
│ └── 02_Modeling.ipynb
├── images/
├── requirements.txt
└── README.md


---

## 👩‍💻 Autora

Proyecto desarrollado por **Nabila Isabel Padilla Resendiz**  
💼 Ingeniera en Mecatrónica | 📊 Aspirante a Data Scientist | 💡 Apasionada por el aprendizaje y la tecnología  

🔗 [LinkedIn](https://www.linkedin.com/in/nabilap/)  
💻 [GitHub](https://github.com/mecatronabi)


