import tensorflow as tf
from keras import utils as utls
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


imageRows, imageCols, cores = 32, 32, 3 
batchSize = 64
numClasses = 10
epochs = 5  


(XTreino, yTreino), (XTeste, yTeste) = cifar10.load_data()


XTreino = XTreino / 255.0
XTeste = XTeste / 255.0


yTreino = utls.to_categorical(yTreino, numClasses)
yTeste = utls.to_categorical(yTeste, numClasses)


nomeDosRotulos = ["Avião", "Automóvel", "Pássaro", "Gato", "Cervo", "Cachorro", "Rato", "Sapo", "Cavalo", "Navio"]


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão")
    plt.show()


def train_and_evaluate_model(model, model_name):
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(XTreino, yTreino, batch_size=batchSize, epochs=epochs, validation_data=(XTeste, yTeste))
    

    predicao = model.predict(XTeste)
    pred_classes = predicao.argmax(axis=1)
    y_true = yTeste.argmax(axis=1)
    

    print(f"Relatório de Classificação para {model_name}:")
    print(classification_report(y_true, pred_classes, target_names=nomeDosRotulos))
    

    plot_confusion_matrix(y_true, pred_classes, nomeDosRotulos)
    

    plt.plot(history.history['accuracy'], 'o-')
    plt.plot(history.history['val_accuracy'], 'x-')
    plt.legend(['Acurácia no Treinamento', 'Acurácia na Validação'], loc='lower right')
    plt.title(f'Treinamento e Validação - Acurácia por Época ({model_name})')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.show()


def build_simple_convnet():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(imageRows, imageCols, cores)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(numClasses, activation='softmax')
    ])
    model.summary()
    return model


print("Treinando o modelo ConvNet simplificado para CIFAR-10:")
train_and_evaluate_model(build_simple_convnet(), "ConvNet CIFAR-10")
