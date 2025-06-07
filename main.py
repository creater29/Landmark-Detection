import kagglehub

path = kagglehub.dataset_download("confirm/google-landmark-dataset-v2-micro")

print("Path to dataset files:", "/content")

import os

# List contents of the dataset folder
print("Files in dataset folder:")
print(os.listdir(path))

for fname in os.listdir(path):
    print("Full path:", os.path.join(path, fname))

import numpy as np
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt
import os
import random
from PIL import Image

samples = 20000
df = pd.read_csv("/content/google-landmark-dataset-v2-micro/gldv2_micro/train.csv")
df = df.loc[:samples,:]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)


print("Size of training data:", df.shape)
print("Number of unique classes:", num_classes)


data = pd.DataFrame(df['landmark_id'].value_counts())
#index the data frame
data.reset_index(inplace=True)
data.columns=['landmark_id','count']

print(data.head(10))
print(data.tail(10))


print(data['count'].describe())#statistical data for the distribution
plt.hist(data['count'],100,range = (0,944),label = 'test')#Histogram of the distribution
plt.xlabel("Amount of images")
plt.ylabel("Occurences")

print("Amount of classes with five and less datapoints:", (data['count'].between(0,5)).sum())

print("Amount of classes with with between five and 10 datapoints:", (data['count'].between(5,10)).sum())

# Sort the unique landmark_ids before using them as bins
sorted_unique_landmarks = sorted(df["landmark_id"].unique())
n = plt.hist(df["landmark_id"], bins=sorted_unique_landmarks)
freq_info = n[0]

plt.xlim(0,data['landmark_id'].max())
plt.ylim(0,data['count'].max())
plt.xlabel('Landmark ID')
plt.ylabel('Number of images')

from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()
lencoder.fit(df["landmark_id"])

def encode_label(lbl):
    return lencoder.transform(lbl)

def decode_label(lbl):
    return lencoder.inverse_transform(lbl)

def get_image_from_number(num):
    fname, label = df.loc[num,:]
    fname = fname + ".jpg"
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    path = os.path.join(f1,f2,f3,fname)
    im = cv2.imread(os.path.join(base_path,path))
    return im, label

print("4 sample images from random classes:")
fig=plt.figure(figsize=(16, 16))

base_path = '/content/google-landmark-dataset-v2-micro/gldv2_micro/images'

for i in range(1,5):
    # Check if the base_path exists before trying to list its contents
    if not os.path.isdir(base_path):
        print(f"Error: The directory '{base_path}' does not exist.")
        # You might want to break the loop or handle this error differently
        break
    a = random.choices(os.listdir(base_path), k=3)
    folder = os.path.join(base_path, a[0], a[1], a[2]) # Use os.path.join for creating paths
    # Check if the folder exists before trying to list its contents
    if not os.path.isdir(folder):
        print(f"Warning: The directory '{folder}' does not exist. Skipping.")
        continue # Skip to the next iteration if the folder doesn't exist
    random_img = random.choice(os.listdir(folder))
    img = np.array(Image.open(os.path.join(folder, random_img))) # Use os.path.join
    fig.add_subplot(1, 4, i)
    plt.imshow(img)
    plt.axis('off')

plt.show()

from keras.applications import VGG19
from keras.layers import *
from keras import Sequential

### Parameters
# learning_rate   = 0.0001
# decay_speed     = 1e-6
# momentum        = 0.09

# loss_function   = "sparse_categorical_crossentropy"
source_model = VGG19(weights=None)
#new_layer = Dense(num_classes, activation=activations.softmax, name='prediction')
drop_layer = Dropout(0.5)
drop_layer2 = Dropout(0.5)


model = Sequential()
for layer in source_model.layers[:-1]: # go through until last layer
    if layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
#     if layer == source_model.layers[-3]:
#         model.add(drop_layer)
# model.add(drop_layer2)
model.add(Dense(num_classes, activation="softmax"))
model.summary()


opt1 = keras.optimizers.RMSprop(learning_rate = 0.0001, momentum = 0.09)
opt2 = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=opt1,
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

#sgd = SGD(lr=learning_rate, decay=decay_speed, momentum=momentum, nesterov=True)
# rms = keras.optimizers.RMSprop(lr=learning_rate, momentum=momentum)
# model.compile(optimizer=rms,
#               loss=loss_function,
#               metrics=["accuracy"])
# print("Model compiled! \n")


### Function used for processing the data, fitted into a data generator.
def get_image_from_number(num, df):
    fname, label = df.iloc[num,:]
    fname = fname + ".jpg"
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    path = os.path.join(f1,f2,f3,fname)
    full_path = os.path.join(base_path,path)
    # Check if the image file exists before reading
    if not os.path.exists(full_path):
        print(f"Error: Image file not found at {full_path}")
        return None, label # Return None for the image if not found

    im = cv2.imread(full_path)
    return im, label

def image_reshape(im, target_size):
    # Check if the image is valid before resizing
    if im is None or im.size == 0:
        return None # Return None or handle the error as appropriate
    return cv2.resize(im, target_size)

def get_batch(dataframe,start, batch_size):
    image_array = []
    label_array = []

    end_img = start+batch_size
    if end_img > len(dataframe):
        end_img = len(dataframe)

    for idx in range(start, end_img):
        n = idx
        im, label = get_image_from_number(n, dataframe)

        # Check if image was loaded successfully and resized
        if im is not None:
            im_resized = image_reshape(im, (224, 224))
            if im_resized is not None:
                image_array.append(im_resized / 255.0)
                label_array.append(label)
            else:
                 print(f"Warning: Skipping image at index {idx} due to resizing failure.")
        else:
            print(f"Warning: Skipping image at index {idx} due to loading failure.")


    label_array = encode_label(label_array)
    return np.array(image_array), np.array(label_array)

batch_size = 16
epoch_shuffle = True
weight_classes = True
epochs = 15

# Split train data up into 80% and 20% validation
train, validate = np.split(df.sample(frac=1), [int(.8*len(df))])
print("Training on:", len(train), "samples")
print("Validation on:", len(validate), "samples")

for e in range(epochs):
    print("Epoch: ", str(e+1) + "/" + str(epochs))
    if epoch_shuffle:
        train = train.sample(frac = 1)
    for it in range(int(np.ceil(len(train)/batch_size))):

        X_train, y_train = get_batch(train, it*batch_size, batch_size)

        # Only train if there are valid images in the batch
        if len(X_train) > 0:
            model.train_on_batch(X_train, y_train)
        else:
            print(f"Warning: Skipping batch {it} in epoch {e+1} as no valid images were loaded.")


model.save("Model.h5")

### Test on training set
batch_size = 16

errors = 0
good_preds = []
bad_preds = []

# Check if the validation set is not empty
if len(validate) > 0:
    for it in range(int(np.ceil(len(validate)/batch_size))):

        X_train, y_train = get_batch(validate, it*batch_size, batch_size)

        # Check if the batch is not empty before predicting
        if len(X_train) > 0:
            result = model.predict(X_train)
            cla = np.argmax(result, axis=1)
            for idx, res in enumerate(result):
                print("Class:", cla[idx], "- Confidence:", np.round(res[cla[idx]],2), "- GT:", y_train[idx])
                if cla[idx] != y_train[idx]:
                    errors = errors + 1
                    # Need to adjust index for bad_preds and good_preds
                    # The index should be relative to the start of the validation dataframe
                    actual_idx_in_validate = it * batch_size + idx
                    bad_preds.append([actual_idx_in_validate, cla[idx], res[cla[idx]]])
                else:
                    actual_idx_in_validate = it * batch_size + idx
                    good_preds.append([actual_idx_in_validate, cla[idx], res[cla[idx]]])
        else:
             print(f"Warning: Skipping prediction for batch {it} as no valid images were loaded.")

    print("Errors: ", errors, "Acc:", np.round(100*(len(validate)-errors)/len(validate),2))

    #Good predictions
    # Ensure good_preds is not empty before processing
    if len(good_preds) > 0:
        good_preds = np.array(good_preds)
        # Sorting needs to happen after converting to numpy array
        good_preds = np.array(sorted(good_preds.tolist(), key = lambda x: x[2], reverse=True)) # Convert to list for sorting
        # Convert back to numpy array if needed, or just work with the list

        fig=plt.figure(figsize=(16, 16))
        # Take min of 5 or the number of good predictions
        num_display = min(5, len(good_preds))
        for i in range(num_display):
            n = int(good_preds[i,0])
            # Fetch the image using the index within the 'validate' dataframe
            img, lbl = get_image_from_number(n, validate)
            # Ensure image was loaded successfully
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fig.add_subplot(1, num_display, i + 1) # Adjust subplot index
                plt.imshow(img)
                lbl2 = np.array(int(good_preds[i,1])).reshape(1,1)
                # To get sample count for the original landmark_id in the full dataset
                original_landmark_id = validate.iloc[n]['landmark_id']
                sample_cnt = list(df.landmark_id).count(original_landmark_id)
                plt.title("Label: " + str(original_landmark_id) + "\nClassified as: " + str(decode_label(lbl2)) + "\nSamples in class " + str(original_landmark_id) + ": " + str(sample_cnt))
                plt.axis('off')
            else:
                print(f"Warning: Could not display image for good prediction at index {n}.")
        if num_display > 0:
            plt.show()
    else:
        print("No good predictions to display.")

    # Display bad predictions similarly if needed
    if len(bad_preds) > 0:
        bad_preds = np.array(bad_preds)
        # You can add code here to sort and display bad predictions if desired
        # For example: bad_preds = np.array(sorted(bad_preds.tolist(), key = lambda x: x[2], reverse=True))
        # Then loop through bad_preds to display images
        print("\nBad predictions:")
        # Example: print first few bad predictions details
        for i in range(min(5, len(bad_preds))):
             n = int(bad_preds[i, 0])
             classified_as = int(bad_preds[i, 1])
             confidence = round(bad_preds[i, 2], 2)
             original_landmark_id = validate.iloc[n]['landmark_id']
             print(f"Index in validate: {n}, Classified as: {decode_label(np.array([classified_as]).reshape(1,1))[0]}, Confidence: {confidence}, GT: {original_landmark_id}")


else:
    print("Validation set is empty. Skipping testing on validation set.")
