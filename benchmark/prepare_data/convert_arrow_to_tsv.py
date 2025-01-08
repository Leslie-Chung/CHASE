# %%
import pyarrow.feather as pf
import pandas as pd
import numpy as np

targets = ['laion1m', 'laion100']
for target in targets:
   table = pf.read_table(f'resources/data/laion/{target}.arrow') 
   df = table.to_pandas()

   def parse_and_convert_array(array):
      s = np.array2string(array, separator=', ', floatmode='fixed', sign='+')
      s = s.replace('\n', '')
      return s
   df['vec'] = df['vec'].apply(parse_and_convert_array)

   def convert_column(column_name, dtype, fill_value=None):
      if fill_value is not None:
         df[column_name] = df[column_name].fillna(fill_value)
      df[column_name] = df[column_name].astype(dtype)
      if fill_value is not None:
         df[column_name] = df[column_name].replace(fill_value, np.nan)
   convert_column('sample_id', 'Int64')
   convert_column('height', pd.Int32Dtype(), -1)
   convert_column('width', pd.Int32Dtype(), -1)
   convert_column('similarity', 'float64')

   df['text'].replace('\t', ' ', inplace=True, regex=True)
   df['url'].replace('\t', ' ', inplace=True, regex=True)
   df.to_csv(f'benchmark/prepare_data/data/{target}_pgvector.tsv', sep='\t', index=False)

   import subprocess
   command = ["sed", "-e", "s/\\[/\\{/g", "-e", "s/\\]/\\}/g", f"benchmark/prepare_data/data/{target}_pgvector.tsv"]
   with open(f"benchmark/prepare_data/data/{target}_vbase.tsv", "w") as output_file:
      subprocess.run(command, stdout=output_file)



