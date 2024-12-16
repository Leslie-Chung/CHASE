import os
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pyarrow.feather as pf


base_path = './'

# read metadata
meta_path = os.path.join(base_path, 'part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet')
meta = pq.read_table(meta_path)

# Ensure that the NSFW column only contains the types 'UNSURE', 'UNLIKELY', and 'NSFW'
replacement_values = ['UNSURE', 'UNLIKELY', 'NSFW']
mask = pc.invert(pc.is_in(meta['NSFW'], value_set=pa.array(replacement_values)))
num_replacements = pc.count(mask)
random_replacements = np.random.choice(replacement_values, size=num_replacements.as_py())
new_nsfw_column = pc.if_else(mask, pa.array(random_replacements), meta['NSFW'])
meta = meta.set_column(meta.schema.get_field_index('NSFW'), 'NSFW', new_nsfw_column)
print(meta)


arrays = []
for i in range(11):
   data = np.load(os.path.join(base_path, f'img_emb_{i}.npy'))
   arrays.append(data)
vectors = np.concatenate(arrays, axis=0)
num_rows = len(vectors)
dimension = len(vectors[0])

# %%
cols_name = ['SAMPLE_ID', 'URL', 'TEXT', 'HEIGHT', 'WIDTH', 'NSFW', 'similarity', 'vec']
result = []
offset1 = 0
length1 = 1000448 * 7
offset2 = 1000448 * 8
length2 = 1000448 * 3
new_meta = pa.concat_tables([meta.slice(offset1, length1),meta.slice(offset2, length2)])
for i in range(len(cols_name) - 1):
   result.append(new_meta[cols_name[i]])
result.append(pa.array(vectors, type=pa.list_(pa.float32(), dimension))) 
cols_name_lower = [col.lower() for col in cols_name]
table = pa.table(result, names=cols_name_lower)
print(table)

# Ensure that the sample_id column is unique
sample_id_column = table.column('sample_id')
unique_sample_ids = pc.unique(sample_id_column)
first_occurrence_indices = {}
for i in range(len(sample_id_column)):
    sample_id = sample_id_column[i].as_py()
    if sample_id not in first_occurrence_indices:
        first_occurrence_indices[sample_id] = i
unique_indices = list(first_occurrence_indices.values())
unique_table = table.take(unique_indices)
t = unique_table['sample_id']
mask = t.is_null().to_numpy()
mask = ~mask
unique_table = unique_table.filter(mask)

# %%
pf.write_feather(unique_table, 'chase/resources/data/laion/laion10m.arrow', 'uncompressed')  
M = 1000000
pf.write_feather(unique_table.slice(0, M), 'chase/resources/data/laion/laion1m.arrow', 'uncompressed')
pf.write_feather(unique_table.slice(M, 100), 'chase/resources/data/laion/laion100.arrow', 'uncompressed') 

# %%
import hnswlib
ids = np.arange(len(vectors))
p = hnswlib.Index(space = 'ip', dim = dimension) # possible options are l2, cosine or ip
p.init_index(max_elements = num_rows, ef_construction = 200, M = 16)
p.add_items(vectors, ids, num_threads = 32)
p.save_index("chase/resources/data/laion/laion1m.vec.hnsw")
