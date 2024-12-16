
import pandas as pd
import chase
from .base import Base
# {"QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime", "total"}
# tmpvenv/bin/python3 benchmark/lingodb-v/run.py --sql_type=topk-filter --user_dbname=lingodb-v
printOrder = ["QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime"]

class CHASE(Base):
   def __init__(self, user_dbname, sql_type, selectivity):
      super().__init__(user_dbname, sql_type, selectivity)
   
   def run(self):
      with open(self.sql_file_path, 'r') as f:
         con = chase.connect_to_db("./resources/data/laion")
         result_df = con.sql(f.read()).to_pandas()
         total_time = 0
         for order in printOrder:
            total_time += con.get_time(order)
         # print(total_time)
         execute_time = con.get_time('executionTime')
         # print(f"Recall = {recall:.3f}, ExecuteTime = {execute_time:.2f} ms, Total = {total_time:.2f} ms")
         
         recall = self.get_recall(result_df)

         self.append_to_csv(recall, execute_time, total_time)

   