from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagenerator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

for img_no in range (1,6):
    img = load_img(path='/home/sadabrata/transfer_learning_opencv/files/comp_images_2/'+str(img_no)+'.jpeg')
    x = img_to_array(img)
    # input data in `NumpyArrayIterator` should have rank 4
    x = x.reshape((1,) + x.shape)
    i = 1
    for batch in datagenerator.flow(x, batch_size=1, save_to_dir='/home/sadabrata/transfer_learning_opencv/dataset/test/sadabrata/', save_format='jpg'):
        i+=1
        if i>40:
            break
print("Collecting Samples Complete")
