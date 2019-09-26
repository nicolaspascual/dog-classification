import datetime
out_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

img_rows, img_cols, channels = 224, 224, 3
input_shape = (img_rows, img_cols, channels)

from load_data import load_data
train_generator, valid_generator, test_generator = load_data('./data/', img_rows, img_cols)

from load_model import load_model
model = load_model(input_shape)

#Start training
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1
)
#Evaluate the model with test set
score = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_VALID)
print('test loss:', score[0])
print('test accuracy:', score[1])

from utils import plot_accuracy, plot_loss, save_model

plot_accuracy(history, f'./result/{out_file_name}')
plot_loss(history, f'./result/{out_file_name}')
save_model(model, f'./out/{out_file_name}/')