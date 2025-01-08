# -*- coding: utf-8 -*-
import psycopg2
import time
import re
import pandas as pd
from .base import Base
import subprocess

log_file = "/u02/pgdata/13/pg_log/postgresql-Thu.log"
# venv/bin/python3 benchmark/postgres/run.py --sql_type=topk-filter --user_dbname=vbase
class PostgresLike(Base):
   def __init__(self, user_dbname, sql_type, selectivity, explain=False, use_index=True):
      super().__init__(user_dbname, sql_type, selectivity)
      dbs = {
         'vbase' : {
            'db': 'laion_vbase',
            # 'host': '172.17.0.2',
            # 'usr': 'hqdb',
         }, 
         'pgvector' : {
            'db': 'laion_pgvector',
            # 'host': '172.17.0.2',
            # 'usr': 'hqdb',
         }, 
         'pase' : {
            'db': 'laion_pase',
            # 'host': '172.17.0.2',
            # 'usr': 'hqdb',
         }, 
      }

      db = dbs[user_dbname]['db']
      # usr = dbs[user_dbname]['usr']
      # host = dbs[user_dbname]['host']
      self.conn = psycopg2.connect(database=db, user='hqdb', password='hqdb', host='localhost')
      
      self.explain = explain
      self.use_index = use_index
      
   def __del__(self):
      self.conn.close()
      # print("destroyed")

   def extract_all_warnings(self):
      input_file = log_file
      grep_command = f"grep -E '(ip times in index|additional sort in vbase|Branch misses|Cache misses|Branches|L1-D misses|L1-I misses|L2 misses|LLC misses|L1-D access|L1-I access|L2 access|LLC access)' {input_file}"
      grep_process = subprocess.Popen(grep_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      grep_output, grep_error = grep_process.communicate()

      if grep_output:
         awk_command = """
         awk -F'=' '
         /ip times/{print "IP times in index: "$2}
         /additional sort/{print "Additional sort in vbase: "$2}
         /Branch misses/{print "Branch misses: "$2}
         /Cache misses/{print "Cache misses: "$2}
         /Branches/{print "Branches: "$2}
         /L1-D misses/{print "L1-D misses: "$2}
         /L1-I misses/{print "L1-I misses: "$2}
         /L2 misses/{print "L2 misses: "$2}
         /LLC misses/{print "LLC misses: "$2}
         /L1-D access/{print "L1-D access: "$2}
         /L1-I access/{print "L1-I access: "$2}
         /L2 access/{print "L2 access: "$2}
         /LLC access/{print "LLC access: "$2}
         '
         """
         awk_process = subprocess.Popen(awk_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
         awk_output, awk_error = awk_process.communicate(input=grep_output)

         print(awk_output.decode())
      
      with open(input_file, 'w') as f:
         f.truncate(0)      
      
   def get_cal_times(self):
      input_file = log_file
      grep_command = f"grep -E '(ip times in index|additional sort in vbase)' {input_file}"
      grep_process = subprocess.Popen(grep_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      grep_output, grep_error = grep_process.communicate()

      if grep_output:
         awk_command = "awk -F'=' '/ip times/{print \"IP times in index: \"$2} /additional sort/{print \"Additional sort in vbase: \"$2}'"
         awk_process = subprocess.Popen(awk_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
         awk_output, awk_error = awk_process.communicate(input=grep_output)

         print(awk_output.decode())

      # with open(input_file, 'w') as f:
      #    f.truncate(0)
         
   def get_perf(self):
      input_file = log_file
      
      grep_command = f"grep -E '(Branch misses|Cache misses|Branches)' {input_file}"
      grep_process = subprocess.Popen(grep_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      grep_output, grep_error = grep_process.communicate()

      if grep_output:
         awk_command = """
         awk -F'=' '/Branch misses/{print "Branch misses: "$2} 
         /Cache misses/{print "Cache misses: "$2} 
         /Branches/{print "Branches: "$2}'
         """
         awk_process = subprocess.Popen(awk_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
         awk_output, awk_error = awk_process.communicate(input=grep_output)

         print(awk_output.decode())
         with open(input_file, 'w') as f:
            f.truncate(0)
      
   def warm_up(self, cursor):
      if self.user_dbname == 'pgvector':
         cursor.execute(f"SET hnsw.ef_search = 49;")
         cursor.execute(f"select pg_prewarm('laion1m', 'buffer', 'main', 0, 71493);")
         cursor.execute(f"select pg_prewarm('laion100', 'buffer', 'main', 0, 7);")
         cursor.execute(f"select pg_prewarm('laion1m_hnsw_pgvector', 'buffer', 'main', 0, 74305);")
      else:
         cursor.execute(f"select pg_prewarm('laion1m', 'buffer', 'main', 0, 142864);")
         cursor.execute(f"select pg_prewarm('laion100', 'buffer', 'main', 0, 14);")
         
   def get_load_index_time(self):
      for notice in self.conn.notices:
         if 'WARNING' in notice:
            match = re.search(r'WARNING:  LoadIndex\(\) took (\d+\.\d+) ms to execute', notice)
            if match:
               # print(notice)
               return float(match.group(1))
      return 0

   def get_execute_time(self, cursor, sql):
      get_time_sql = 'EXPLAIN (ANALYSE, TIMING, BUFFERS) ' + sql
      cursor.execute(get_time_sql)
      time_info = cursor.fetchall()
      load_index_time = self.get_load_index_time() 
      # print(time_info)
      match = re.search(r'Execution Time: ([\d.]+) ms', time_info[-1][0])
      if match:
         execution_time = float(match.group(1))
         if execution_time < load_index_time:
            load_index_time = 0
         return execution_time - load_index_time
      return 0
   
   def prepare_sql(self):
      with open(self.sql_file_path, 'r') as file:
         sql_content = file.read()
         if self.explain:
            sql_content = 'EXPLAIN ANALYSE ' + sql_content
         if self.user_dbname != 'pgvector':
            sql_content = sql_content.replace('[', '{').replace(']', '}').replace('<#>', '<*>')
            
            if self.user_dbname == 'vbase' and self.use_index:
               # if self.sql_type == 'range' or self.sql_type == 'range-filter':
               pattern = re.compile(r'\}\'\s*<\s*([-?0-9.]+)')
               match = pattern.search(sql_content)
               if match:
                  number = match.group(1)
                  # number = number.lstrip('-')
                  sql_content = sql_content.replace('{', '{' + f'{number}, ').replace('<*>', '<<*>>')
                  sql_content = pattern.sub('}\'', sql_content)
                        
               # if self.sql_type == 'distance-join' or self.sql_type == 'distance-join-filter':
               pattern = r"([\w.]+)\s*<\*\>\s*([\w.]+)\s*<\s*([-?0-9.]+)"
               if re.search(pattern, sql_content):
                  def replacer(match):
                        table1, table2, num = match.groups()[:3]
                        return f"{table1} <<*>> array_cat(ARRAY[cast({num} as float8)], {table2})"
                  sql_content = re.sub(pattern, replacer, sql_content)
      return sql_content
   
   def get_gt(self):
      return pd.read_csv(self.gt_path, sep='\t', index_col=0)
   
   def execute_sql(self):
      cursor = self.conn.cursor()
      
      if self.use_index:
         cursor.execute(f"SET enable_seqscan = OFF;")
         cursor.execute(f"SET enable_indexscan = ON;")
         cursor.execute("""
         set max_parallel_workers = 32;
         set max_parallel_workers_per_gather = 32;
         set max_parallel_maintenance_workers = 32;
         set min_parallel_table_scan_size = 0;
                  """)
      else:
         cursor.execute(f"SET enable_indexscan = OFF;")

      sql = self.prepare_sql()
      self.warm_up(cursor)
      
      execute_time = self.get_execute_time(cursor, sql)
      
      with open(log_file, 'w') as f:
         f.truncate(0)
      cursor.execute(sql)
      self.get_cal_times()
      # self.extract_all_warnings()
      self.conn.commit()
      # self.get_perf()
      results = cursor.fetchall()
      column_names = [desc[0] for desc in cursor.description]
      # execute_time = 1
      cursor.close()
      return results, column_names, execute_time
      # return 1, 1, 1
   
   def run(self):
      results, column_names, total_time = self.execute_sql()
      # return
      if self.explain:
         print(results)
         pattern = re.compile(r'^.*Index Scan using.*$', re.MULTILINE)
         for res in results:
            matches = pattern.findall(res[0])
            for match in matches:
               print(match)
      else:
         result_df = pd.DataFrame(results, columns=column_names)
         execute_time = total_time
         
         # print(f"Recall = {1:.3f}, ExecuteTime = {execute_time:.2f} ms, Total = {total_time:.2f} ms")
         
         recall = self.get_recall(result_df)
         
         self.append_to_csv(recall, execute_time, total_time)
      # subprocess.run(['rm', '-rf', '/u02/pgdata/13/pg_log/*'])
