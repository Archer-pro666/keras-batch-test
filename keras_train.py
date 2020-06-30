from keras import optimizers
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),padding='SAME',activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),padding='SAME',activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),padding='SAME',activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(256,(3,3),padding='SAME',activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(512,(3,3),padding='SAME',activation='relu'))
model.add(layers.Conv2D(512,(3,3),padding='SAME',activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(1024,(3,3),padding='SAME',activation='relu'))
model.add(layers.Conv2D(1024,(3,3),padding='SAME',activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2560,activation='relu'))
model.add(layers.Dense(4,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])

train_dir='D:/train random'
validation_dir='D:/validation222'
test_dir='D:/test'
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='categorical')
validation=test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='categorical')
test=test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='categorical')

history=model.fit_generator(train,steps_per_epoch=1947,epochs=20,validation_data=validation,validation_steps=(19461/20))
model.save('D:/keras/模型/cell.h5')
print('模型已保存')

test_loss,test_acc=model.evaluate_generator(test,steps=(88+49+3936+2844)/20)
print('test_loss:',test_loss)
print('test_acc:',test_acc)

acc=history.history['acc']
loss=history.history['loss']
val_acc=history.history['val_acc']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='train_acc')
plt.plot(epochs,val_acc,'b',label='val_acc')
plt.xlabel('epoch',fontsize=10.5)
plt.ylabel('acc',fontsize=10.5)
plt.title('train and validation acc')
plt.legend()
plt.savefig('D:/keras/result/acc_cell.jpg')

plt.figure()
plt.plot(epochs,loss,'bo',label='train_loss')
plt.plot(epochs,val_loss,'b',label='val_loss')
plt.xlabel('epoch',fontsize=10.5)
plt.ylabel('loss',fontsize=10.5)
plt.title('train and validation loss')
plt.legend()
plt.savefig('D:/keras/result/loss_cell.jpg')
plt.show()
