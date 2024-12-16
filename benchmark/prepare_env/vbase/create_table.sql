create database laion_vbase;
\c laion_vbase
create extension vectordb;
CREATE EXTENSION pg_prewarm;
CREATE EXTENSION pg_buffercache;
CREATE TABLE laion1m (
   sample_id int8 PRIMARY KEY,
   url text,
   text text,
   height int4,
   width int4,
   nsfw text,
   similarity float8,
   vec float8[512] NOT NULL
);
copy laion1m from '/laion/laion1m_vbase.tsv' DELIMITER E'\t' csv quote e'\x01' HEADER;
create index laion1m_hnsw_vbase on laion1m using hnsw(vec hnsw_vector_inner_product_ops) with(dimension=512,distmethod=inner_product);
CREATE TABLE laion100 (
   sample_id int8 PRIMARY KEY,
   url text,
   text text,
   height int4,
   width int4,
   nsfw text,
   similarity float8,
   vec float8[512] NOT NULL
);
copy laion1m from '/laion/laion100_vbase.tsv' DELIMITER E'\t' csv quote e'\x01' HEADER;



create database laion_pase;
\c laion_pase
create extension vectordb;
CREATE EXTENSION pg_prewarm;
CREATE EXTENSION pg_buffercache;
CREATE TABLE laion1m (
   sample_id int8 PRIMARY KEY,
   url text,
   text text,
   height int4,
   width int4,
   nsfw text,
   similarity float8,
   vec float8[512] NOT NULL
);
copy laion1m from '/laion/laion1m_vbase.tsv' DELIMITER E'\t' csv quote e'\x01' HEADER;
create index laion1m_hnsw_pase on laion1m using pase_hnsw(vec pase_hnsw_vector_inner_product_ops) with(dimension=512,distmethod=inner_product);
CREATE TABLE laion100 (
   sample_id int8 PRIMARY KEY,
   url text,
   text text,
   height int4,
   width int4,
   nsfw text,
   similarity float8,
   vec float8[512] NOT NULL
);
copy laion1m from '/laion/laion100_vbase.tsv' DELIMITER E'\t' csv quote e'\x01' HEADER;
