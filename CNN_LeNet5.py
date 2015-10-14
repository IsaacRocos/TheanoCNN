# coding=utf-8
import os
import sys
import timeit
import numpy
import cPickle
import theano
import theano.tensor as T

from LogisticRegresion_SGD import LogisticRegression, load_data
from MLPerceptron import HiddenLayer
from CNN.ConvPoolLayer import LeNetConvPoolLayer


def evaluateLeNet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20, 50], batch_size=500):
    """ Red neuronal convolutiva (LeNet5)
    :type learning_rate: float
    :param learning_rate: tasa de aprendzaje (factor para el gradiente stocastico)

    :type n_epochs: int
    :param n_epochs: numero máximo de empocas para ejecutar el optimizador

    :type dataset: string
    :param dataset: path para el set usado en  entrenamiento/prueba

    :type nkerns: lista de enteros
    :param nkerns: numero de kernels en cada capa
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    print "Tamanio de batch: ", batch_size
    print "#Batches Entrenamiento: ", n_train_batches
    print "#Batches Prueba: ", n_test_batches
    print "#Batches Validacion: ", n_valid_batches


    # Asignar variables simbolicas para los datos
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # los datos son presentados como imagenes rasterizadas
    y = T.ivector('y')  # las etiquetas se presentan como vectodes 1D de [int]

    ###########################
    # CONSTRUCCION DEL MODELO #
    ###########################
    print '... Construyendo Modelo.'

    # Remodela la matriz de imagenes rasterizadas de forma (batch_size, 28 * 28)
    # a un tensor 4D, compatible con la clase LeNetConvPoolLayer
    # (28, 28) es el tamanio de las imagenes en MNIST.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Primer capa de pooling convolutiva
    # el filtrado reduce el tamanio de la imagen a (28-5+1 , 28-5+1) = (24, 24)
    # ademas con maxpooling la imagen se reduce a  (24/2, 24/2) = (12, 12)
    # se tiene un tensor 4D de salida de forma  (batch_size, nkerns[0], 12, 12)
    l0_layerInputCNN = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    l1_layerInputCNN = LeNetConvPoolLayer(
        rng,
        input=l0_layerInputCNN.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = l1_layerInputCNN.output.flatten(2)


    # construct a fully-connected sigmoidal layer
    l2_FCHiddenLayer = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=l2_FCHiddenLayer.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + l2_FCHiddenLayer.params + l1_layerInputCNN.params + l0_layerInputCNN.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###################
    # ENTRENAR MODELO #
    ###################
    print '... Entrenando'
    # parametros de paro anticipado
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # esperar ma's cuando una mejora se presenta
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'Entrenando @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            print minibatch_index,']',cost_ij, ","

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('Epoca %i, minibatch %i/%i, error de validation  %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoca %i, minibatch %i/%i, error de prueba del'
                           'mejor modelo %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimización completa.')
    print('Mejor puntaje de validación %f %% obtenido en la iteración %i, '
          'con desempeño de prueba %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('El entrenamiento de  ' +
                          os.path.split(__file__)[1] +
                          ' se ejecutó durante %.2fmins' % ((end_time - start_time) / 60.))



def save_Model_Status(train_set_x, train_set_y, valid_set_x,valid_set_y):
    save_file = open('ModelStatus')
    cPickle.dump(train_set_x.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    cPickle.dump(train_set_y.get_value(borrow=True), save_file, -1)  # .. and it triggers much more efficient
    cPickle.dump(valid_set_x.get_value(borrow=True), save_file, -1)  # .. storage than numpy's default
    cPickle.dump(valid_set_y.get_value(borrow=True), save_file, -1)  # .. storage than numpy's default
    save_file.close()

def load_Model_Status(train_set_x, train_set_y, valid_set_x,valid_set_y):
    save_file = open('ModelStatus')
    train_set_x.set_value(cPickle.load(save_file), borrow=True)
    train_set_y.set_value(cPickle.load(save_file), borrow=True)
    valid_set_x.set_value(cPickle.load(save_file), borrow=True)
    valid_set_y.set_value(cPickle.load(save_file), borrow=True)


if __name__ == '__main__':
    evaluateLeNet5()


def experiment(state, channel):
    evaluateLeNet5(state.learning_rate, dataset=state.dataset)

