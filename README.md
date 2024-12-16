

# CHASE: Designing A Novel Framework for Hybrid Query Processing in Query Compilation

## Installation

### Prerequisites

Ensure that the following dependencies are installed on your system:

1. **Python 3.10 or higher**
2. **Standard build tools**, including `cmake` and `Ninja`
3. **Clang++ 16 or higher**

### Installation Steps

1. Clone the CHASE repository and navigate to the project directory:

   ```shell
   cd chase
   ```

2. Set up a Python virtual environment and install required packages:

   ```shell
   python3 -m venv venv
   venv/bin/pip install -r requirements.txt
   venv/bin/python3 -c "import pyarrow; pyarrow.create_library_symlinks()"
   ```

### Building CHASE

#### Debug Version

To build the debug version of CHASE, run:

```shell
make build-debug
```

#### Release Version

To build the release version of CHASE, run:

```shell
make build-release
```

### Building the Python Package

To build the Python package for CHASE, execute the following script:

```shell
tools/python/bridge/create_package.sh
```

With these steps completed, CHASE should be properly installed and ready for use.





## Getting Started

### Initialization

1. Create the sample table:

   ```sql
   -- /sql_path/create_sample.sql
   CREATE TABLE sample (
       id int8 PRIMARY KEY, 
       embedding vector(3) NOT NULL
   );
   ```

   ```shell
   build/chase-debug/run-sql /sql_path/create_sample.sql /data_path/
   ```

2. Insert sample data:

   ```
   -- /sql_path/insert_samples.sql
   INSERT INTO sample (id, embedding) 
   VALUES (1, '[1,2,3]'), (2, '[4,5,6]');
   ```

   ```shell
   build/chase-debug/run-sql /sql_path/insert_samples.sql /data_path/
   ```

3. Create an HNSW index:

   ```sql
   -- /sql_path/create_hnswindex.sql
   CREATE INDEX ON sample USING hnsw (embedding vector_l2_ops) 
   WITH (m = 16, ef_construction = 200);
   ```

   ```shell
   build/chase-debug/run-sql /sql_path/create_hnswindex.sql /data_path/
   ```

   **Note:** Currently, only HNSW indexes are supported with two similarity metrics:

   - **vector_l2_ops:** L2 distance without square root.
   - **vector_innerproduct_ops:** Negative inner product.

   **Index Parameters:**

   - `m` - Maximum number of connections per layer.
   - `ef_construction` - Size of the dynamic candidate list for constructing the graph.

### Querying

1. Retrieve the top 5 nearest neighbors based on L2 distance:

   ```sql
   -- /sql_path/q1.sql
   SELECT * FROM sample ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
   ```

2. Filter by `id` and retrieve nearest neighbors:

   ```sql
   -- /sql_path/q2.sql
   SELECT * FROM sample WHERE id = 1
   ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
   ```

3. execute the query

   ```shell
   build/chase-debug/run-sql /sql_path/q2.sql /data_path/
   ```

   

## Reproduce Experiments in the Paper

### Create Docker

#### pgvector

```shell
# get pgvector
git clone --branch v0.7.3 https://github.com/pgvector/pgvector.git
cd pgvector
cp chase/benchmark/prepare_env/pgvector/Dockerfile .
cp -r chase/benchmark/prepare_env/scripts .
cp benchmark/prepare_env/pgvector/hnswscan.c src/

# build docker
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t pgvector -f Dockerfile .

# run docker
docker run --privileged --shm-size=100g --name=pgvector -e PGPASSWORD=pgvector -e PGUSERNAME=pgvector -e PGDATABASE=pgvector -v chase/benchmark/prepare_data:/laion pgvector 
```



#### vbase & pase

```shell
# get vbase
git clone https://github.com/microsoft/MSVBASE.git
cd MSVBASE
git submodule update --init --recursive
./scripts/patch.sh

cp chase/include/runtime/HNSW/space_ip.h thirdparty/hnsw/hnswlib
cp chase/include/runtime/HNSW/space_l2.h thirdparty/hnsw/hnswlib
cp chase/benchmark/prepare_env/vbase/hnswindex_builder.cpp src/
cp chase/benchmark/prepare_env/vbase/hnswindex.cpp src/
cp chase/benchmark/prepare_env/vbase/pase_hnswindex.cpp src/

cp chase/benchmark/prepare_env/vbase/Dockerfile .
cp -r chase/benchmark/prepare_env/scripts .

# build docker
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t vbase -f Dockerfile .
docker run --privileged --shm-size=100g --name=vbase -e PGPASSWORD=vectordb -e PGUSERNAME=vectordb -e PGDATABASE=vectordb -v chase/benchmark/prepare_data:/laion vbase 
```



#### chase & lingodb

```shell
docker cp chase vbase:/home/postgres
```



### Generate Data

```shell
cd chase
./benchmark/prepare_data/generate_data.sh
```



### Create Table & index

#### pgvector

```shell
docker cp chase/benchmark/prepare_env/pgvector/create_table.sql vbase:/
docker exec -it pgvector bash
psql -U vectordb -h localhost -d laion_pgvector -f /create_table.sql
```



#### vbase & pase

```shell
docker exec -it vbase bash
psql -U vectordb -h localhost -d laion_vbase -f chase/benchmark/prepare_env/vbase/create_table.sql
```



#### chase & lingodb

```shell
docker exec -it vbase bash
cd chase
echo "CREATE TABLE laion1m (
   sample_id int8 PRIMARY KEY,
   url text,
   text text,
   height int4,
   width int4,
   nsfw text,
   similarity float8,
   vec vector(512) NOT NULL
);" > init.sql

build/chase-debug/run-sql init.sql resources/data/laion
cp resources/data/laion/laion1m.metadata.json resources/data/laion/laion100.metadata.json
```



### run benchmark

```shell
docker exec -it vbase bash
cd chase
./bench.sh
```

