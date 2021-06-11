import io
import os
import sys
import time
import json
import numpy as np
import sqlite3
import tensorflow as tf


con = sqlite3.connect(r'C:\Users\adamm\Documents\Programs\data\paperdoll\data\chictopia/chictopia.sqlite3')
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
# cur_brands.execute("""
#     SELECT * FROM tags WHERE type = 'Brand'
# """)

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
""")


clothing = cur_clothing.fetchall()
colors = cur_color.fetchall()
# print("Clothing Types = " + str(len(clothing)))
# print("Color Types = " + str(len(colors)))
# print("Brand Types = " + str(len(cur_brands.fetchall())))


alltypes = []

for color in colors:
    for type in clothing:
        garment = (str(color[0]), str(type[0]))
        alltypes += [garment]
alltypes.append((".","."))

alltypes_json = json.dumps(alltypes)

alltypes_file = open("type.json", "w")
alltypes_file.write(alltypes_json)
alltypes_file.close()

# print(alltypes[:5])
# print("There are {} types".format(len(alltypes)), "\n")
print("Fetching outfit data ...")
results = np.array(cur_outfits.fetchall())
print("Fetched outfit data.")

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
# print(encoded_outfits[:5])


seq_length = 4

outfit_dataset = tf.data.Dataset.from_tensor_slices(encoded_outfits)


# for i in outfit_dataset.take(5):
#   print(i)

sequences = outfit_dataset

# for item in sequences.take(5):
#     print(id2garment[item.numpy()])

def split_input_target(chunk):
    input_clothes = chunk[:-1]
    target_clothes = chunk[1:]
    return input_clothes, target_clothes

dataset = sequences.map(split_input_target)

# for input_example, target_example in  dataset.take(1):
#   print ('Input data: ',  id2garment[input_example.numpy()])
#   print ('Target data: ', id2garment[target_example.numpy()])
#   for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print("Step {:4d}".format(i))
#     print("  input: {} ({:s})".format(input_idx, repr(id2garment[input_idx])))
#     print("  expected output: {} ({:s})".format(target_idx, repr(id2garment[target_idx])))

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# print(dataset)

# Length of the vocabulary in chars
vocab_size = len(alltypes)

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

model = build_model(
  vocab_size = len(alltypes),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

# for input_example_batch, target_example_batch in dataset.take(1):
#   example_batch_predictions = model(input_example_batch)
#   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#
# print(sampled_indices)
#
# print("Input: \n", id2garment[input_example_batch[0].numpy()])
# print()
# print("Next Clothes Predictions: \n", [id2garment[index] for index in sampled_indices])

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())
#

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=30
print("Started training ...")
start_of_training = time.time()
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
end_of_training = time.time()
print("Training complete in {}".format(end_of_training-start_of_training))

embed_layer = model.layers[0]
weights = embed_layer.get_weights()[0]
print(weights.shape)
# print(weights[0])
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, item in enumerate(alltypes):
  vec = weights[num]
  out_m.write(str(item) + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
