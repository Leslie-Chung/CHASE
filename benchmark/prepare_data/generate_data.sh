for ((i=0; i<=10; i++)) do 
  wget -P benchmark/prepare_data/data "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/img_emb_${i}.npy"
done

# for ((i=0; i<=10; i++)) do 
#   wget "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/metadata/metadata_${i}.parquet"
# done

wget -P benchmark/prepare_data/data https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet

mkdir resoucres/data/laion
python3 benchmark/prepare_data/create_arrow_data_from_LAION.py
python3 benchmark/prepare_data/convert_arrow_to_tsv.py
