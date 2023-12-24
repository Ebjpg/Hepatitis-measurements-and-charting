import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

filename = "hepatitis.csv"
# Türkçe formatındaki dosyayı okurken ayraç ve desimal işaretini belirt
data = pd.read_csv(filename, sep=';', decimal='.')

sns.pairplot(data)
plt.show()

corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Veriyi özellikler (X) ve etiketler (y) olarak ayıran kısım
X = data.drop('class', axis=1)
y = data['class']

# Veriyi eğitim ve test setine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sınıflandırıcı modelini seç ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)




