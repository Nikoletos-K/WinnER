import itertools 
import matplotlib.pyplot as plt
import numpy as np

# Function that creates a confusion matrix
def create_ConfusionMatrix(confusionMatrix,title):
    plt.figure(figsize = (8,5))
    classes = ['Different','Same']
    cmap = plt.cm.Blues
    plt.grid(False)
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusionMatrix.max() / 2.
    for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
        plt.text(j, i, confusionMatrix[i, j],horizontalalignment="center",color="white" if confusionMatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.ylim([1.5, -.5])
    plt.show()  