import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Carregando os dados
dados = pd.read_csv('prouni_2005_2019.csv',nrows=200000)

# Selecionando as colunas relevantes
colunas = [ 'ANO_CONCESSAO_BOLSA', 'NOME_IES_BOLSA', 
           'MODALIDADE_ENSINO_BOLSA', 'NOME_CURSO_BOLSA', 'NOME_TURNO_CURSO_BOLSA', 
            'SEXO_BENEFICIARIO_BOLSA', 'RACA_BENEFICIARIO_BOLSA', 
            'BENEFICIARIO_DEFICIENTE_FISICO', 
           'REGIAO_BENEFICIARIO_BOLSA', 'SIGLA_UF_BENEFICIARIO_BOLSA', 
           'MUNICIPIO_BENEFICIARIO_BOLSA', 'idade']

# Selecionando os recursos (X) e o alvo (y)
X = dados[colunas]
y = dados['TIPO_BOLSA']

# Convertendo variáveis categóricas em variáveis dummy
X = pd.get_dummies(X, drop_first=True)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando o classificador de árvore de decisão
clf = DecisionTreeClassifier()

# Treinando o classificador
clf.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
y_pred = clf.predict(X_test)

# Calculando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)


accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)

# Plotando a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.show()
