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
