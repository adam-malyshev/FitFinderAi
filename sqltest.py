import sqlite3
import numpy as np
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

con = sqlite3.connect(r'C:\Users\adamm\Documents\Programs\data\paperdoll\data\chictopia/chictopia.sqlite3')
cur = con.cursor()
cur2 = con.cursor()
cur3 = con.cursor()

cur.execute("""
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

cur2.execute("""
    SELECT name FROM tags WHERE type == "Clothing";
""")

cur3.execute("""
    SELECT name FROM tags WHERE type == "Color";
""")





# print(header(cur2.description, 15))
clothing = cur2.fetchall()
# print(results)
# for row in clothing:
#     print(table(row,15))

print("Clothing items: " + str(len(clothing)), "\n")

# print(header(cur3.description, 15))
colors = cur3.fetchall()
# print(results)
# for row in colors:
#     print(table(row,15))

print("Color items: " + str(len(colors)), "\n")


alltypes = []

for color in colors:
    for type in clothing:
        garment = (str(color[0]), str(type[0]))
        alltypes += [garment]
alltypes.append((".","."))

print(alltypes[:5])
print("There are {} types".format(len(alltypes)), "\n")

results = np.array(cur.fetchall())

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
