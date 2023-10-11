# DL


























URL Source for Dataset
MNIST Handwritten digit dataset
https://www.kaggle.com/competitions/digit-recognizer/data
COVID-19 Chest X-Ray dataset
 https://www.kaggle.com/datasets/bachrr/covid-chest-xray
Pneumonia Dataset
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Breast Cancer Dataset-
https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
Image Augmentation Dataset
 https://www.kaggle.com/datasets/yasserhessein/shoulder-implant-xray
Lane Detection Dataset
https://www.kaggle.com/datasets/tryingit/roadlane-detection-evaluation-2013
Fingerprint Verification
 https://www.kaggle.com/datasets/peace1019/fingerprint-dataset-for-fvc2000-db4-b
Language Translator Dataset
 https://www.kaggle.com/datasets/umasrikakollu72/hindi-english-truncated-corpus
Face Recognition Dataset
 https://www.kaggle.com/datasets/hereisburak/pins-face-recognition 
Face Mask Detection Dataset
https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
Flower-17 dataset- 
https://www.kaggle.com/datasets/saidakbarp/17-category-flowers
Sentiment Analysis Dataset- 
https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert


Working with Google Colab
Loading a dataset 
# Mount Google Drive (if your dataset is stored there)
from google.colab import drive
drive.mount('/content/drive')
# Specify the path to your dataset
dataset_path = '/content/drive/MyDrive/dataset_folder/'
# List the contents of the dataset folder
import os
os.listdir(dataset_path)


















Implement Deep Feed Forward Neural Network model to classify MNIST Handwritten Digit Recognition dataset. 
Make sure you have TensorFlow installed (pip install tensorflow).
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # Normalize pixel values to [0, 1]
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 input images
    Dense(128, activation='relu'),   # Fully connected layer with 128 units and ReLU activation
    Dense(64, activation='relu'),    # Fully connected layer with 64 units and ReLU activation
    Dense(10, activation='softmax') ])  # Output layer with 10 units for 10 classes and softmax activation
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
Implement ShallowNet Convolution Neural Network Model using Stochastic Gradient Descent-(ADAM Optimizer) for Healthcare Analysis
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
# Define paths to your dataset directories
train_data_dir = 'path_to_train_data_directory'
test_data_dir = 'path_to_test_data_directory'
# Define image dimensions and other parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 10
num_classes = 2  # Number of classes (COVID-19 positive and negative)
# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Data augmentation for testing images (only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
# Load and preprocess the dataset using data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
# Build the ShallowNet CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')])
# Compile the model with the Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(train_generator, epochs=epochs, validation_data=test_generator)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")

Implement Generative Adversarial Deep Learning model for Image Generation and Data Augmentation. 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
# Define parameters
random_dim = 100
adam = Adam(lr=0.0002, beta_1=0.5)
# Create the generator model
generator = Sequential([
    Dense(256, input_dim=random_dim, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])
# Create the discriminator model
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=adam)
# Combine generator and discriminator to create GAN
discriminator.trainable = False
gan_input = Input(shape=(random_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=adam)
# Load and preprocess the dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 255.0
x_train = np.expand_dims(x_train, axis=-1)
# Training parameters
batch_size = 64
epochs = 10000
sample_interval = 1000
# Training the GAN
for e in range(epochs + 1):
    # Train discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    noise = np.random.normal(0, 1, (batch_size, random_dim))
    fake_imgs = generator.predict(noise)
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # Train generator
    noise = np.random.normal(0, 1, (batch_size, random_dim))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)
    if e % sample_interval == 0:
        print(f"Epoch {e}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")
        # Save generated images
        samples = 10
        noise = np.random.normal(0, 1, (samples, random_dim))
        generated_imgs = generator.predict(noise)
        generated_imgs = generated_imgs.reshape(-1, 28, 28)
        plt.figure(figsize=(10, 2))
        for i in range(samples):
            plt.subplot(1, samples, i + 1)
            plt.imshow(generated_imgs[i], cmap='gray')
            plt.axis('off')
        plt.show()


Implement bagging based Ensemble Deep CNN model for automated Lane Detection and Assistance System for self-Driven cars
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
# Load and preprocess the lane detection dataset
# Split the dataset into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
num_base_models = 5
ensemble_models = []
for i in range(num_base_models):
    # Create a base CNN model
    base_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
        # Add more convolutional layers and flatten
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Adjust for lane markings classes
    ])
    # Compile the base model
    base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the base model on a bootstrap sample
    sampled_indices = np.random.choice(len(x_train), size=len(x_train), replace=True)
    base_model.fit(x_train[sampled_indices], y_train[sampled_indices], epochs=num_epochs, batch_size=batch_size)
    # Store the trained base model in the ensemble
    ensemble_models.append(base_model)
# Aggregate predictions from the ensemble
ensemble_predictions = []
for model in ensemble_models:
    predictions = model.predict(x_val)
    ensemble_predictions.append(predictions)
# Aggregate the ensemble predictions (e.g., averaging)
aggregated_predictions = np.mean(ensemble_predictions, axis=0)
# Lane assistance logic
# Evaluate the ensemble on the validation set
ensemble_loss, ensemble_accuracy = ensemble_models[0].evaluate(x_val, y_val)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

Implement the fingerprint verification system using Deep Convolution Neural Network model.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define data directories
train_data_dir = 'path_to_train_data_directory'
test_data_dir = 'path_to_test_data_directory'
# Define image dimensions and other parameters
img_width, img_height = 128, 128
batch_size = 32
epochs = 20
# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Data augmentation for testing images (only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
# Load and preprocess the dataset using data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(train_generator, epochs=epochs, validation_data=test_generator)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")

Implement an automated language translator using Recurrent Neural Network (Sequence-To-Sequence model) with TensorFlow Deep Learning Library.
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

# Prepare example data (replace with your dataset)
english_sentences = ['I am a student.', 'He is a teacher.']
french_sentences = ['Je suis étudiant.', 'Il est enseignant.']
# Tokenize sentences and create vocabulary
english_vocab = set(" ".join(english_sentences).split())
french_vocab = set(" ".join(french_sentences).split())
english_vocab_size = len(english_vocab)
french_vocab_size = len(french_vocab)
# Create word-to-index and index-to-word mappings
en_word2idx = {w: i for i, w in enumerate(english_vocab)}
fr_word2idx = {w: i for i, w in enumerate(french_vocab)}
fr_idx2word = {i: w for i, w in enumerate(french_vocab)}
# Convert sentences to sequences of integer indices
encoder_input_data = [[en_word2idx[word] for word in sentence.split()] for sentence in english_sentences]
decoder_input_data = [[fr_word2idx[word] for word in sentence.split()] for sentence in french_sentences]
# Define model parameters
latent_dim = 256  # Hidden state size of LSTM
# Define encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(english_vocab_size, latent_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]
# Define decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(french_vocab_size, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(french_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model (replace with your data)
model.fit([encoder_input_data, decoder_input_data], np.zeros_like(decoder_input_data), epochs=10)
# Create inference models for translation
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# Perform translation
def translate_sentence(input_sentence):
    input_seq = [en_word2idx[word] for word in input_sentence.split()]
    input_seq = np.array(input_seq)
    input_seq = np.expand_dims(input_seq, axis=0)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = fr_word2idx['<start>']  # Start token
    translated_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = fr_idx2word[sampled_token_index]
        translated_sentence += ' ' + sampled_word
        if sampled_word == '<end>' or len(translated_sentence.split()) > 50:  # End token or max length
            break
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return translated_sentence
# Translate English sentence to French
input_sentence = 'They are students.'
translated_sentence = translate_sentence(input_sentence)
print('Input:', input_sentence)
print('Translation:', translated_sentence)





Implement a Computer Vision Based Face Recognition System using one-shot learning InceptionV3 CNN Model.
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
# Load InceptionV3 model pre-trained on ImageNet
base_model = InceptionV3(weights='imagenet', include_top=False)
# Add a global average pooling layer to the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)
# Preprocess an input image and generate an embedding
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img
# Load and preprocess reference and query images
reference_image_path = 'path_to_reference_image.jpg'
query_image_path = 'path_to_query_image.jpg'
reference_embedding = feature_extractor.predict(preprocess_image(reference_image_path))
query_embedding = feature_extractor.predict(preprocess_image(query_image_path))
# Calculate similarity score using cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2.T)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity
# Calculate similarity between reference and query embeddings
similarity_score = cosine_similarity(reference_embedding, query_embedding)
# Set a similarity threshold for recognition
threshold = 0.8
if similarity_score >= threshold:
    print("Recognized!")
else:
    print("Not recognized.")

Implement the Computer Vision Model for face mask detection using Faster RCNN an object detection architecture
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import cv2
# Load model pipeline configuration
configs = config_util.get_configs_from_pipeline_file('path_to_pipeline_config.pbtxt')
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
# Load checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('path_to_checkpoint').expect_partial()
# Load label map
label_map_path = 'path_to_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
# Load and preprocess image for inference
image_path = 'sample_image.jpg'
image_np = cv2.imread(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]
# Perform inference
detections = detection_model(input_tensor)
# Visualize and display results
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np,
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(np.uint32),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=10,
    min_score_thresh=0.3,
    agnostic_mode=False)

cv2.imshow('Face Mask Detection', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()

Implement the state-of-the-art transfer learning CNN models for multiple class flower image classification and perform the comparative analysis to identify the best model.
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# Load and preprocess the dataset
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    'path_to_train_directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    'path_to_validation_directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
# Build InceptionV3 model with transfer learning
base_model_inception = InceptionV3(weights='imagenet', include_top=False)
x = base_model_inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model_inception = Model(inputs=base_model_inception.input, outputs=predictions)
# Build ResNet50 model with transfer learning
base_model_resnet = ResNet50(weights='imagenet', include_top=False)
x = base_model_resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions)
# Compile the models
model_inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the models
history_inception = model_inception.fit(train_generator, epochs=10, validation_data=validation_generator)
history_resnet = model_resnet.fit(train_generator, epochs=10, validation_data=validation_generator)

Perform Sentiment Analysis in Natural Language Processing using RNN-Long Short Term Memory (LSTM) architecture with “Bag of Words” dictionary.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# Sample dataset
positive_reviews = ["I love this product!", "Great experience!", "Highly recommended."]
negative_reviews = ["Terrible product.", "Waste of money.", "Not happy with it."]
reviews = positive_reviews + negative_reviews
labels = [1, 1, 1, 0, 0, 0]  # Positive = 1, Negative = 0
# Tokenize text
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
# Pad sequences
max_sequence_length = 10  # Choose an appropriate value
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
# Create LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=16, input_length=max_sequence_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Convert labels to numpy array
labels = np.array(labels)
# Split data into train and test sets
train_size = int(0.8 * len(padded_sequences))
x_train, x_test = padded_sequences[:train_size], padded_sequences[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]
# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
