import io
import os
import sys
import time
import json
import numpy as np
import sqlite3
import tensorflow as tf
import random
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
type_file = open("type.json", "r")
alltypes = type_file.read()
alltypes = json.loads(alltypes)
type_file.close()
dt = np.dtype([('color', np.unicode_, 16), ('type', np.unicode_, 16)])

newtypes = []
for type in alltypes:
    newtypes.append(tuple(type))

id2garment = np.array(newtypes, dt)
# print(id2garment[:5])
garment2id = {u:i for i,u in enumerate(newtypes)}
# print(garment2id)
# Length of the vocabulary in types
vocab_size = len(newtypes)

# The embedding dimension
embedding_dim = 378

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


checkpoint_dir = './training_checkpoints'

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

print(model.summary())


def generate_outfit(model, start_clothes_array):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 11

  # Converting our start string to numbers (vectorizing)
  input_eval = [garment2id[s] for s in start_clothes_array]
  # print(input_eval)
  input_eval = tf.expand_dims(input_eval, 0)
  # print(input_eval)
  # Empty string to store our results
  outfit_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # print(predicted_id)
      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      outfit_generated.append(id2garment[predicted_id])
      # print(outfit_generated)
      outarr = start_clothes_array + outfit_generated
  return (outarr)

print("\n")
outfit = generate_outfit(model, start_clothes_array = [('black', 't-shirt')])
print("Generated outfit",outfit)
print("\n")
print("\n")
vecs_tsv = open("vecs.tsv")
read_tsv = csv.reader(vecs_tsv, delimiter="\t")
id2embedding = []
for row in read_tsv:
    id2embedding.append([eval(value) for value in row])
embedding2id = {tuple(u):i for i,u in enumerate(id2embedding)}
# print(id2embedding[0])


dt = np.dtype([('color', np.unicode_, 16), ('type', np.unicode_, 16)])
id2garment = np.array(newtypes, dt)
# print(id2garment[:5])
garment2id = {u:i for i,u in enumerate(newtypes)}

# for type in newtypes[-10:]:
#     print("{}:{} \n".format(newtypes.index(type), type))

random.seed(time.time() ** 25)
def genRandomWardrobe(length):
    wardrobe = []
    for _ in range(length):
        rand_index = random.randint(0, newtypes.index(newtypes[-1]))
        wardrobe.append(id2garment[rand_index])
    return wardrobe
mywardrobe = genRandomWardrobe(1000)
# mywardrobe = alltypes
encoded_wardrobe = [garment2id[tuple(garment)] for garment in mywardrobe]
Embededd_wardrobe = [id2embedding[garment] for garment in encoded_wardrobe]
encoded_outfit = [garment2id[tuple(garment)] for garment in outfit if tuple(garment) != (".",".")]
Embedded_outfit = [id2embedding[garment] for garment in encoded_outfit]
# print("Wardrobe",mywardrobe)
# print("\n")
# print("\n")
# print(encoded_wardrobe)
# print(embededd_wardrobe[0])
# print(encoded_outfit)
# print(embedded_outfit[0])

def cosine_similarity(vector_x, vector_y):
    dotproduct = 0
    euclidian_x = 0
    euclidian_y = 0
    for i in range(len(vector_x)):
        x = vector_x[i]
        y = vector_y[i]
        euclidian_x += x **2
        euclidian_y += y **2
        dotproduct += x * y
    euclidian_x = euclidian_x ** 0.5
    euclidian_y = euclidian_y ** 0.5
    return (dotproduct/(euclidian_x*euclidian_y))
print("sim(<1,2>,<3,4>) is {}".format(cosine_similarity([1,2],[3,4])))
print("sim(<1,2>,<1,2.1>) is {}".format(cosine_similarity([1,2],[1,2.1])))
def nearest_outfit(embedded_outfit, embedded_wardrobe, dim=256):
    output_outfit = []
    for outfit_garment in embedded_outfit:
        distance_dict = {}
        for index, wardrobe_garment in enumerate(embedded_wardrobe):
            distance_dict[index] = cosine_similarity(outfit_garment, wardrobe_garment)
        sorted_distances = sorted(distance_dict.items(), key = lambda x: x[1])
        print("Sorted distances", sorted_distances[-10:],"\n")
        closest_index = sorted_distances[-1][0]
        output_outfit.append(embedded_wardrobe[closest_index])
    return output_outfit

closest_outfit = nearest_outfit(Embedded_outfit, Embededd_wardrobe)
suggested_outfit = [embedding2id[tuple(garment)] for garment in closest_outfit]
suggested_outfit = [id2garment[id] for id in suggested_outfit]
print("Closest match",suggested_outfit)
