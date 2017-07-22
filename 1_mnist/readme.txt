
1.kaggle mnist   https://www.kaggle.com/c/digit-recognizer
    train.csv : train data
    test.csv  : test data

2.tensorflow
    mnist_tf.py   : tensorflow code, score:0.99014
    load_mnist.py : read train and test data

3.keras
    mnist_keras.py : keras code, score: 0.98557,using ImageDataGenerator to get higher score;

4.Resnet50
    mnist_res.py   : keras Resnet50 code, out of memory using GPU, need to optimize;
    cvt_img2rgb.py : convert mnist image to RGB image for Resnet50, the input shape is (224,224,3)
    figures_save.py: save train loss and acc to figure 