import argparse
from dbs import GroundTruth, CHASE, PostgresLike
import time

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Process some command line arguments.')
   parser.add_argument('--sql_type', type=str, required=True, help='The type of SQL to execute')
   parser.add_argument('--selectivity', type=float, required=True)
   parser.add_argument('--dbnames', nargs='+', required=True, help='The names of the databases')
   parser.add_argument('--useindex', action='store_false', required=False)
   parser.add_argument('--explain', action='store_true', required=False)
   parser.add_argument('--times', type=int, required=True)
   args = parser.parse_args()
   
   dbnames = args.dbnames
   print(f"SQL_Type = {args.sql_type}, Selectivity = {args.selectivity}")
   times = args.times
   for dbname in dbnames:
      for i in range(times):
         db = None
         print(f"{dbname} tries to run")
         time.sleep(2)
         
         if dbname == 'lingodb':
            db = GroundTruth(dbname, args.sql_type, args.selectivity)
         elif dbname == 'chase': 
            db = CHASE(dbname, args.sql_type, args.selectivity)
         elif dbname == 'vbase' or dbname == 'pgvector' or dbname == 'pase':
            db = PostgresLike(dbname, args.sql_type, args.selectivity, args.explain, args.useindex)
         
         db.run()
         print(f"{dbname} finish!")

   

   