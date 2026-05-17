import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,
ConfusionMatrixDisplay)


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion matrix"):

    labels = np.arange(len(class_names))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
        colorbar=False
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cm


def plot_history(history, title= "Träningskurvor"):
    history_df = pd.DataFrame(history.history)

    epochs = range(1, len(history_df) + 1)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, history_df["loss"], label = "Training loss")
    plt.plot(epochs, history_df["val_loss"], label = "Validation loss")
    plt.xlabel("Epok")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df["accuracy"], label="Training accuracy")
    plt.plot(epochs, history_df["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epok")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def calculate_per_classs_accuracy(cm, class_names):

    support = cm.sum(axis=1)
    correct = np.diag(cm)

    accuracy = np.divide(
        correct,
        support,
        out=np.zeros_like(correct, dtype=float),
        where=support != 0
    )

    result = pd.DataFrame({
        "class_id": np.arange(len(class_names)),
        "class_name": class_names,
        "support": support,
        "correct": correct,
        "accuracy": accuracy
    })

    return result.sort_values("accuracy")

def compare_histories(histories, labels):

    plt.figure(figsize=(14,5))

    # LOSS
    plt.subplot(1,2,1)

    for history, label in zip(histories, labels):

        plt.plot(
            history.history["val_loss"],
            label=f"{label}"
        )

    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


    # ACCURACY
    plt.subplot(1,2,2)

    for history, label in zip(histories, labels):

        plt.plot(
            history.history["val_accuracy"],
            label=f"{label}"
        )

    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()