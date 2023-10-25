import csv
from client import Client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import data_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1 - Coleta de dados
print("--------------------------------------------------")
print("1 - Coleta de dados\n")

file = open('../resources/clients.csv', 'r')
reader = csv.reader(file)
reader.__next__()

clients = []
cont: int = 0

# 2 - Limpeza e preparação dos dados
print("--------------------------------------------------")
print("2 - Limpeza e preparação de dados\n")

for row in reader:
    c = Client(
        row.__getitem__(0),
        row.__getitem__(1),
        data_utils.convert_to_boolean(row.__getitem__(2)),
        data_utils.convert_to_boolean(row.__getitem__(3)),
        data_utils.convert_to_boolean(row.__getitem__(4)),
        int(row.__getitem__(5)),
        data_utils.convert_to_boolean(row.__getitem__(6)),
        data_utils.convert_to_boolean(row.__getitem__(7)),
        data_utils.convert_to_boolean(row.__getitem__(8)),
        data_utils.convert_to_boolean(row.__getitem__(9)),
        data_utils.convert_to_boolean(row.__getitem__(10)),
        data_utils.convert_to_boolean(row.__getitem__(11)),
        data_utils.convert_to_boolean(row.__getitem__(12)),
        data_utils.convert_to_boolean(row.__getitem__(13)),
        data_utils.convert_to_boolean(row.__getitem__(14)),
        row.__getitem__(15),
        data_utils.convert_to_boolean(row.__getitem__(16)),
        row.__getitem__(17),
        data_utils.convert_to_float(row.__getitem__(18)),
        data_utils.convert_to_float(row.__getitem__(19)),
        data_utils.convert_to_boolean(row.__getitem__(20))
    )

    clients.append(c)
    cont += 1

client_data = {
    'COSTUMER_ID': [c.customer_id for c in clients],
    'GENDER': [c.gender for c in clients],
    'SENIOR_CITIZEN': [c.senior_citizen for c in clients],
    'PARTNER': [c.partner for c in clients],
    'DEPENDENTS': [c.dependents for c in clients],
    'TENURE': [c.tenure for c in clients],
    'PHONE_SERVICE': [c.phone_service for c in clients],
    'MULTIPLE_LINES': [c.multiple_lines for c in clients],
    'INTERNET_SERVICE': [c.internet_service for c in clients],
    'ONLINE_SECURITY': [c.online_security for c in clients],
    'ONLINE_BACKUP': [c.online_backup for c in clients],
    'DEVICE_PROTECTION': [c.device_protection for c in clients],
    'TECH_SUPPORT': [c.tech_support for c in clients],
    'STREAMING_TV': [c.streaming_tv for c in clients],
    'STREAMING_MOVIES': [c.streaming_movies for c in clients],
    'CONTRACT': [c.contract for c in clients],
    'PAPERLESS_BILLING': [c.paperless_billing for c in clients],
    'PAYMENT_METHOD': [c.payment_method for c in clients],
    'MONTHLY_CHARGES': [c.monthly_charges for c in clients],
    'TOTAL_CHARGES': [c.total_charges for c in clients],
    'CHURN': [c.churn for c in clients],
}

df = pd.DataFrame(client_data)
print(df.head())
print("Total de linhas: " + str(len(df)))

# 3 - Análise Exploratória de Dados (EDA)
print("--------------------------------------------------")
print("3 - Análise Exploratória de Dados (EDA)\n")

print("\n------MONTHLY_CHARGES---------")
print("Media:", df['MONTHLY_CHARGES'].mean())
print("Moda:", df['MONTHLY_CHARGES'].mode()[0])
print("Mediana:", df['MONTHLY_CHARGES'].median())
print("Máximo:", df['MONTHLY_CHARGES'].max())
print("Mínimo:", df['MONTHLY_CHARGES'].min())
print("Desvio Padrão:", df['MONTHLY_CHARGES'].std())

print("\n-----TOTAL_CHARGES-------------")
print("Media:", df['TOTAL_CHARGES'].mean())
print("Moda:", df['TOTAL_CHARGES'].mode()[0])
print("Mediana:", df['TOTAL_CHARGES'].median())
print("Máximo:", df['TOTAL_CHARGES'].max())
print("Mínimo:", df['TOTAL_CHARGES'].min())
print("Desvio Padrão:", df['TOTAL_CHARGES'].std())

# 4 - Modelagem preditiva e estatística
print("--------------------------------------------------")
print("4 - Modelagem preditiva e estatística")
print("[Gráfico]")

monthly_charges = df['MONTHLY_CHARGES']
total_charges = df['TOTAL_CHARGES']
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)

plt.title("Média - Monthly Charges")
plt.hist(monthly_charges, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
plt.axvline(monthly_charges.mean(), color='k', linestyle='dashed', linewidth=1)
plt.subplot(2, 2, 2)

plt.title("Média - Total Charges")
plt.hist(total_charges, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
plt.axvline(total_charges.mean(), color='k', linestyle='dashed', linewidth=1)
plt.subplot(2, 2, 3)

plt.title("Mínimo, Máximo, Mediana - Monthly Charges")
plt.boxplot(monthly_charges, vert=False, widths=0.5)
plt.text(monthly_charges.median(), 1.1, 'Mediana', color='r', fontsize=12, ha='center')
plt.text(monthly_charges.min(), 1.2, 'Mínimo', color='b', fontsize=12, ha='center')
plt.text(monthly_charges.max(), 1.2, 'Máximo', color='g', fontsize=12, ha='center')
plt.subplot(2, 2, 4)

plt.title("Mínimo, Máximo, Mediana - Total Charges")
plt.boxplot(total_charges, vert=False, widths=0.5)
plt.text(total_charges.median(), 1.1, 'Mediana', color='r', fontsize=12, ha='center')
plt.text(total_charges.min(), 1.2, 'Mínimo', color='b', fontsize=12, ha='center')
plt.text(total_charges.max(), 1.2, 'Máximo', color='g', fontsize=12, ha='center')
plt.subplot(2, 2, 1)

plt.legend(["Média"])
plt.tight_layout()
plt.show()

# 5 - Machine Learning - Aplicando algoritmo supervisionado 'LogisticRegression'
print("--------------------------------------------------")
print("5 - Machine Learning - Aplicando algoritmo supervisionado 'LogisticRegression'\n")

X = df[['TENURE',
        'MONTHLY_CHARGES',
        'TOTAL_CHARGES',
        'ONLINE_SECURITY',
        'ONLINE_BACKUP',
        'DEVICE_PROTECTION',
        'TECH_SUPPORT',
        'STREAMING_TV',
        'STREAMING_MOVIES']]
Y = df['CHURN']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
confusion = confusion_matrix(Y_test, Y_pred)
classification = classification_report(Y_test, Y_pred)

print("Acurácia:", accuracy)
print("Matriz de Confusão:")
print(confusion)

print("\nRelatório de Classificação:")
print(classification)

# 6 - Visualização de dados - Correlação
print("--------------------------------------------------")
print("6 - Visualização de dados - Correlação")
print("[Gráfico]")

numeric_columns = df.select_dtypes(include=['number', 'boolean'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

# 7 - Tomada de decisão - Gráficos de Pizza - Churn
print("--------------------------------------------------")
print("7 - Tomada de decisão - Gráficos de Pizza - Churn\n")

# Contar a quantidade de churn (1) e não churn (0) nos dados reais
churn_counts = df['CHURN'].value_counts()

# Contar a quantidade de previsões de churn (1) e não churn (0) pelo modelo
predicted_churn_counts = pd.Series(Y_pred).value_counts()

# Preparar os dados para os gráficos de pizza
labels = ['Continuam assinando', 'Deixam de assinar']
sizes = [churn_counts[0], churn_counts[1]]
predicted_sizes = [predicted_churn_counts[0], predicted_churn_counts[1]]
colors = ['green', 'lightcoral']
explode = (0.1, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
ax1.set_title('Distribuição de Churn (Real)')
ax1.axis('equal')

ax2.pie(predicted_sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
ax2.set_title('Distribuição de Previsões de Churn')
ax2.axis('equal')
plt.show()
