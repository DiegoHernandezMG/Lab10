import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def load_data(filepath):
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(filepath)

def evaluate_classifier(model, X, y, validation_method):
    """Evalúa el modelo según el método de validación."""
    if validation_method == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, conf_matrix, y_test, y_pred
    elif validation_method == '10fold':
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        return scores.mean(), None, None, None
    elif validation_method == 'leaveoneout':
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
        return scores.mean(), None, None, None

def plot_confusion_matrix(conf_matrix, labels, title):
    """Grafica la matriz de confusión."""
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def main():
    # Especifica los archivos CSV que vas a usar
    datasets = ["iris/bezdekIris.data", "iris/aves.csv", "iris/fruit.csv"]
    
    for dataset in datasets:
        print(f"Evaluando dataset: {dataset}")
        data = load_data(dataset)
        
        # Ajusta según tus columnas (última columna como target)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Clasificadores
        classifiers = {
            'Naive Bayes': GaussianNB(),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Métodos de validación
        validation_methods = ['holdout', '10fold', 'leaveoneout']
        
        for name, model in classifiers.items():
            print(f"\nClasificador: {name}")
            for method in validation_methods:
                print(f"\nMetodo de validacion: {method}")
                accuracy, conf_matrix, y_test, y_pred = evaluate_classifier(model, X, y, method)
                print(f"Accuracy: {accuracy}")
                
                if conf_matrix is not None:  # Solo aplica para holdout
                    print(f"Matriz de Confusion:\n{conf_matrix}")
                    plot_confusion_matrix(conf_matrix, labels=set(y), title=f"{name} - {method}")
                    
if __name__ == "__main__":
    main()