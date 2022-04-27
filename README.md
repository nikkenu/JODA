# JODA (Johdatus datatieteeseen 2022)

### Running the program

1. This instruction touches only harjoitustyo.ipynb file.
2. First download all libraries/packages. (You can find the list of external libraries from the file itself or from [version-info.txt](https://github.com/nikkenu/JODA/blob/main/harjoitustyo/version-info.txt))
3. Create a Twitter account and enter to Twitter developer dashboard. Next you need to create an Bearer token and copy & paste it respectively in the code.
4. Download all required .csv files: [Training & testing dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and [Extra testing dataset](https://www.kaggle.com/datasets/hassanamin/textdb3?select=fake_or_real_news.csv). Next copy & paste these files to JODA/files/ folder.
5. Now you should be able to run your code. Have fun!

#### In case you do not want to train your model by yourself, follow these extra steps:
1. Uncomment this part of the code:

```python
# model = tf.keras.models.load_model('../model.h5')
```

2. Skip or comment these sections out

```python
model = Sequential()

# Add LSTM layers
model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False))
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
```

```python
history = model.fit(X_train, y_train, validation_split=0.3, epochs=10, batch_size=batch_size, shuffle=True, verbose = 1)
```

3. You can also skip the model testing for quicker runtime.

```python
predict = model.predict_classes(X_test)
print(classification_report(y_test, predict))
```

