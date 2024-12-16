
from .base import Base
import subprocess
import chase
# {"QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime", "total"}
# tmpvenv/bin/python3 benchmark/lingodb/run.py --sql_type=topk-filter --user_dbname=lingodb
printOrder = ["QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime"]

class GroundTruth(Base):
   def __init__(self, user_dbname, sql_type, selectivity):
      super().__init__(user_dbname, sql_type, selectivity)

   def get_gt_mlir(self):
      with open(self.ori_mlir_path, 'w') as f:
         subprocess.run(['./build/chase-debug/sql-to-mlir', self.sql_file_path, './resources/data/laion/'
                        ], stdout=f, cwd='.')
      
   def get_opt_gt_mlir(self):
      with open(self.opt_mlir_path, 'w') as f:
         subprocess.run(['./build/chase-debug/mlir-db-opt', '-allow-unregistered-dialect', self.ori_mlir_path,
                           '-split-input-file', '-mlir-print-debuginfo', '-relalg-query-opt', '-lower-relalg-to-subop', '-lower-subop'
                        ], stdout=f, cwd='.')
      
   def run(self):
      self.get_gt_mlir()
      self.get_opt_gt_mlir()
      with open(self.opt_mlir_path, 'r') as f:
         con = chase.connect_to_db("./resources/data/laion")
         lines = f.read()
         res = con.mlir(lines).to_pandas()
         res.to_csv(self.gt_path, sep='\t', index=True)
         total_time = 0
         for order in printOrder:
            # print(f"{order}: {con.get_time(order)}")
            total_time += con.get_time(order)
         # print(total_time)
         # print(res)
         execute_time = con.get_time('executionTime')
         # print(f"Recall = {1:.3f}, ExecuteTime = {execute_time:.2f} ms, Total = {total_time:.2f} ms")  # 这个时间不对，缺少了部分lower的时间
         self.append_to_csv(1, execute_time, total_time)
   