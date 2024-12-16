create database laion_pgvector;
\c laion_pgvector
create extension vector;
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
   vec vector(512) NOT NULL
);
SET max_parallel_maintenance_workers = 7; -- plus leader
SET maintenance_work_mem = '50GB';
copy laion1m from '/laion/laion1m_pgvector.tsv' DELIMITER E'\t' csv quote e'\x01' HEADER;
create index laion1m_hnsw_pgvector on laion1m using hnsw(vec vector_ip_ops) with(m = 16, ef_construction = 200);


CREATE TABLE laion100 (
   sample_id int8 PRIMARY KEY,
   url text,
   text text,
   height int4,
   width int4,
   nsfw text,
   similarity float8,
   vec vector(512) NOT NULL
);
copy laion1m from '/laion/laion100_pgvector.tsv' DELIMITER E'\t' csv quote e'\x01' HEADER;