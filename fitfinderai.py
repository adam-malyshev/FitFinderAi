import io
import os
import sys
import time
import numpy as np
import lmdb
import sqlite3
import tensorflow as tf


import pandas as pd
from PIL import Image
from IPython.display import display


# class PhotoData(object):
#     def __init__(self, path):
#         self.env = lmdb.open(
#             path, map_size=2**36, readonly=True, lock=False
#         )
#
#     def __iter__(self):
#         with self.env.begin() as t:
#             with t.cursor() as c:
#                 for key, value in c:
#                     yield key, value
#
#     def __getitem__(self, index):
#         key = str(index).encode('ascii')
#         with self.env.begin() as t:
#             data = t.get(key)
#         if not data:
#             return None
#         with io.BytesIO(data) as f:
#             image = Image.open(f)
#             image.load()
#             return image
#
#     def __len__(self):
#         return self.env.stat()['entries']


# path = "file:/Users/adammalyshev/Documents/projects/data/paperdoll/data/chictopia/chictopia.sqlite3?mode=ro"
# db = sqlite3.connect(path, uri=True)
#
# photos = pd.read_sql("""
#     SELECT
#         *,
#         'http://images2.chictopia.com/' || path AS url
#     FROM photos
#     WHERE photos.post_id IS NOT NULL AND file_file_size IS NOT NULL
# """, con=db)


def header(header_tuple, column_width):
    out = ""
    for column in header_tuple:
        out = out + str(column[0])
        out = out + ((column_width - len(str(column[0]))) * " ")
    out = out + "\n"
    return out
def table(row_tuple, column_width):
    out = ""
    for column in row_tuple:
        out = out + str(column)
        out = out + ((column_width - len(str(column))) * " ")
    out = out + "\n"
    return out
# def displayTags(cursor, column_width):
#     print(header(cursor.description, column_width))
#     tag_rows = cursor.fetchall()
#     for i in range(len(tag_rows)):
#         row = list(tag_rows[i])
#         row.append(i)
#         new_row=tuple(row)
#         print(table(new_row,column_width))

con = sqlite3.connect(r'/Users/adammalyshev/Documents/projects/data/paperdoll/data/chictopia/chictopia.sqlite3')
cur = con.cursor()
cur_clothing = con.cursor()
cur_color = con.cursor()
cur_brands = con.cursor()
cur_outfits = con.cursor()


# cur.execute("SELECT * FROM garments WHERE clothing_id IS NOT NULL AND color_id IS NOT NULL AND brand_id IS NOT NULL AND store_id IS NOT NULL AND post_id IS NOT NULL ;")
# cur.execute("SELECT * FROM garments WHERE id < 100")
# cur.execute("SELECT id, chictopia_id, type, style_id, garments, season_id FROM posts WHERE id = 1000")
cur_clothing.execute("""
    SELECT name FROM tags WHERE type = 'Clothing';
""")
cur_color.execute("""
    SELECT name FROM tags WHERE type == "Color";
""")
cur_brands.execute("""
    SELECT * FROM tags WHERE type = 'Brand'
""")

cur_outfits.execute("""
    SELECT
      post_id,
      GROUP_CONCAT(name) AS outfit
    FROM (
      SELECT
        post_id,
        "('" ||
        coalesce(colors.name,"") || "','" ||
        coalesce(clothings.name,"") ||
        "')" AS name,
        clothings.id AS id
      FROM garments
      LEFT OUTER JOIN tags AS clothings
          ON garments.clothing_id = clothings.id
      LEFT OUTER JOIN tags AS colors
          ON garments.color_id = colors.id
      WHERE colors.id != ('') AND clothings.id != ('')
      ) AS garment_tuples
    GROUP BY post_id
    HAVING COUNT(id) <= 12 AND COUNT(id) > 0
    LIMIT 10
""")


clothing = cur_clothing.fetchall()
colors = cur_color.fetchall()
print("Clothing Types = " + str(len(clothing)))
print("Color Types = " + str(len(colors)))
# print("Brand Types = " + str(len(cur_brands.fetchall())))


alltypes = []

for color in colors:
    for type in clothing:
        garment = (str(color[0]), str(type[0]))
        alltypes += [garment]
alltypes.append((".","."))

print(alltypes[:5])
print("There are {} types".format(len(alltypes)), "\n")

results = np.array(cur_outfits.fetchall())

dt = np.dtype([('color', np.unicode_, 16), ('type', np.unicode_, 16)])

garment2id = {u:i for i,u in enumerate(alltypes)}
id2garment = np.array(alltypes, dt)

class Encoder:
    def __init__(self, reference):
        self.reference = reference
    def encode(self, outfits):
        out = []
        for outfit in outfits:
            new_fit = []
            for garment in outfit:
                garment = tuple(garment)
                new_fit.append(self.reference.index(garment))
            out.append(new_fit)
        return out
    def decode(self, outfits):
        return [[self.reference[encoded_garment] for encoded_garment in outfit] for outfit in outfits]

encoder = Encoder(alltypes)


outfits = []

for outfit in results:
    new_outfit = eval(outfit[1])
    if isinstance(new_outfit[0], tuple):
        outfits += [list(new_outfit)]
    else:
        outfits += [[new_outfit]]
max_outfit_length = 12

def normalizer(outfit_array):
    num_blanks = max_outfit_length - len(outfit_array)
    outfit_array += [id2garment[3648]] * num_blanks
    return outfit_array

outfits = list(map(normalizer, outfits))

outfits = np.array(outfits, dt)

encoded_outfits = np.array(encoder.encode(outfits))
print(encoded_outfits[:5])


seq_length = 4

outfit_dataset = tf.data.Dataset.from_tensor_slices(encoded_outfits)


# for i in outfit_dataset.take(5):
#   print(i)

sequences = outfit_dataset

for item in sequences.take(5):
    print(id2garment[item.numpy()])

def split_input_target(chunk):
    input_clothes = chunk[:-1]
    target_clothes = chunk[1:]
    return input_clothes, target_clothes

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ',  id2garment[input_example.numpy()])
  print ('Target data: ', id2garment[target_example.numpy()])
  for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(id2garment[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(id2garment[target_idx])))

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(dataset)

# Length of the vocabulary in chars
vocab_size = len(alltypes)

# The embedding dimension
embedding_dim = 256

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

model = build_model(
  vocab_size = len(alltypes),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

print(sampled_indices)

print("Input: \n", id2garment[input_example_batch[0].numpy()])
print()
print("Next Clothes Predictions: \n", [id2garment[index] for index in sampled_indices])

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10

# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

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

output = [generate_outfit(model, start_clothes_array = ['hat']) for i in range(100)]
for outfit in output:
    print(outfit)


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, item in enumerate(cleaned_types):
  vec = weights[num]
  out_m.write(item + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
