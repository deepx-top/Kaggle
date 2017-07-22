import matplotlib.pyplot as plt
from keras.callbacks import History


def figures(history, figure_name="plots"):
    """ method to visualize accuracies and loss vs epoch for training as well as testind data
        Argumets: history     = an instance returned by model.fit method
                  figure_name = a string representing file name to plots. By default it is set to "plots"
       Usage: hist = model.fit(X,y)
       figures(hist) """
    if isinstance(history, History):
        hist = history.history
        epoch = history.epoch
        acc = hist['acc']
        loss = hist['loss']
        # val_loss = hist['val_loss']
        # val_acc = hist['val_acc']
        plt.figure(1)

        plt.subplot(2, 2, 1)
        plt.plot(epoch, acc)
        plt.title("Training accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(2, 2, 2)
        plt.plot(epoch, loss)
        plt.title("Training loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(2, 2, 3)
        # plt.plot(epoch, val_acc)
        plt.title("Validation Acc vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")

        plt.subplot(2, 2, 4)
        # plt.plot(epoch, val_loss)
        plt.title("Validation loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.tight_layout()
        plt.savefig(figure_name)
    else:
        print("Input Argument is not an instance of class History")