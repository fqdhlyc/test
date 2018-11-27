import datetime
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Lambda,Dropout,Conv2D
import keras
import numpy as np
import random
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_match_pred_vec(Y_pred, Y_label):
    assert len(Y_pred) == len(Y_label)
    Y_pred_class = np.argmax(Y_pred, axis = 1)
    Y_label_class = np.argmax(Y_label, axis = 1)
    return Y_pred_class == Y_label_class

def getmatch(pred,label):
    pred = pred[0]
    Y_pred_class = np.argmax(pred, axis = 0)
    Y_label_class = np.argmax(label, axis = 0)
    return Y_pred_class == Y_label_class
    
def calculate_accuracy(Y_pred, Y_label):
    match_pred_vec = get_match_pred_vec(Y_pred, Y_label)

    accuracy = np.sum(match_pred_vec) / float(len(Y_label))
    return accuracy

def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):

    if input_shape is not None:
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding,
                          input_shape=input_shape)
    else:
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                                strides=strides, padding=padding)

def get_test_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)
    del X_train, y_train
    return X_test, Y_test

def get_val_dataset(self):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
	
    val_size = 5000
    X_val = X_train[:val_size]
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
    X_val = X_val.astype('float32') / 255
    y_val = y_train[:val_size]
    Y_val = np_utils.to_categorical(y_val, 10)
    del X_train, y_train, X_test, y_test
    return X_val, Y_val

def get_model(input_shape, nb_filters, nb_classes, pre_filter):
    model = Sequential()

   
    scaler = lambda x: x

    layers = [Lambda(scaler, input_shape=input_shape)]
    layers += [Lambda(pre_filter, output_shape=input_shape)]

    layers += [Dropout(0.2),
              conv_2d(nb_filters, (8, 8), (2, 2), "same"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Dropout(0.5),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    model.add(Activation('softmax'))

    return model

def create_model():
    pre_filter=lambda x:x
    input_shape = (28, 28, 1)
    nb_filters = 64
    nb_classes = 10
    return get_model(input_shape, nb_filters, nb_classes, pre_filter)

def select_seeds(x,y,model):
    randlist = []
    x_seeds = []
    y_seeds = []
    for i in range(20):
        r = os.urandom(16)
        rand = int(r.encode('hex'),16)%10000
        #rand = int.from_bytes(r,byteorder='little')
        if rand in randlist:
            continue
        else:
            randlist.append(rand)
            test = x[rand]
            test = test.reshape((1,28,28,1))
            result = model.predict(test)
            if (getmatch((result),y[rand])):
                x_seeds.append(x[rand])
                y_seeds.append(y[rand])
    return np.array(x_seeds),np.array(y_seeds)

def get_attack_vec(ep):
    attack = ep*(np.random.randint(3,size=(28,28,1))-1)
    #attack[attack<0.33] = 0
    #attack[(attack>=0.33) & (attack < 0.66)] = 1
    #attack[attack<=0.66] = -1
    #attack = attack*ep
    #attack.resize((28,28,1))
    return attack

def select_seeds_by_class(x,y,model):
    x_seeds = []
    y_seeds = []
    count = [0]*10
    for i in range(10000):
        temp = int(np.argmax(y[i], axis = 0))
        if (count[temp]==10):
            continue
        elif(sum(count)==100):
            break
        else:
            test = x[i]
            test = test.reshape((1,28,28,1))
            result = model.predict(test)
            if (getmatch((result),y[i])):
                #print(temp)
                count[temp]+=1
                x_seeds.append(x[i])
                y_seeds.append(y[i])
    x = np.array(x_seeds[0:10])
    y = np.array(y_seeds[0:10])
    return x,y



if __name__ == '__main__':
    chans = create_model()
    X_test,Y_test = get_test_dataset()
    chans.load_weights("MNIST_cleverhans.keras_weights.h5")
    chans.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
    
    r = os.urandom(4)
    #rand = int(r.encode('hex'),16)
    rand = int.from_bytes(r,byteorder='little')
    #x_seeds,y_seeds = select_seeds(X_test, Y_test,chans)
    x_seeds,y_seeds = select_seeds_by_class(X_test, Y_test,chans)
    #print(x_seeds[0])
    #print(y_seeds[0])
    print(len(x_seeds))
    print(len(y_seeds))
    Y_pred = chans.predict(x_seeds)
    acc_all = calculate_accuracy(Y_pred,y_seeds)
    print('Test accuracy on legitimate examples %.4f' % (acc_all))
    ad_x_index = []
    ad_ep = []
    attack_vector = []
    
    ad_vector=[]
    
    test_vec= np.zeros((10000,28,28,1))
    val_vec = np.zeros((10000,10))
    
    for i in range(len(x_seeds)):
        print("seed: %d"%i)
        print(datetime.datetime.now())
        for z in range(16):
            #ep = (1./255.)*(2**z)
            ep = 0.14+(z*0.01)
            print("ep: %f"%ep)
            for j in range(10000):
                attack = get_attack_vec(ep)
                
                result1 = x_seeds[i]+attack
                
                result1[result1>1] = 1
                result1[result1<0] = 0
                
                test_vec[j] = result1
                val_vec[j] = y_seeds[i]

                #result1 = result1.reshape((1,28,28,1))
                
           
                '''if(not getmatch(chans.predict(result1),y_seeds[i])):
                    ad_x_index.append(i)
                    ad_ep.append(ep)
                    attack_vector.append(result1)
                    ad_vector.append(attack)'''
            y_predicted = chans.predict(test_vec)
            y_predicted = np.argmax(y_predicted,axis=1)
            val = np.argmax(val_vec,axis=1)
            difference = np.where(y_predicted!=val)
            difference = difference[0]
            for k in difference:
                ad_x_index.append(i)
                ad_ep.append(ep)
                attack_vector.append(test_vec[k])
                #ad_vector.append()
            
    

    np.save('x_seeds',x_seeds)
    np.save('y_seeds',y_seeds)
    np.save('ad_x_index',np.array(ad_x_index))
    np.save('ad_ep',np.array(ad_ep))
    np.save('attack_vector',np.array(attack_vector))
    np.save('ad_vector',np.array(ad_vector))

    print(ad_x_index)
    print(ad_ep)

    ind = ad_x_index[0]

    exam = x_seeds[ind]
    
    adver = attack_vector[0]
    adver_result = chans.predict(adver)

    adver_num = np.argmax(adver_result[0],axis=0)
    example_num = np.argmax(y_seeds[0],axis=0)


    example = x_seeds[ind]
    example = example.reshape((28,28))

    adv = adver.reshape((28,28))
    

    example = example * 255.
    example = np.rint(example)
    adv = adv * 255.
    adv = np.rint(adv)
    

    plt.gray()
    plt.imshow(example)
    plt.savefig('%dexample_%d_%d.png'%(rand,rand,adver_num))
    plt.close()
    plt.gray()
    plt.imshow(adv)
    plt.savefig('%dadv_%d_%d.png'%(rand,rand,example_num))
    plt.close()
    
    print("")
    a = np.array(ad_ep)
    unique,counts = np.unique(a,return_counts=True)
    plt.plot(unique,counts,'bo')
    plt.xlabel('epsilon')
    plt.ylabel('adversarial examples')
    plt.savefig('%dTesting%d.png'%(rand,rand))
    #plt.show()
    exit()
