import pandas as pd
import os

class Base:
   def __init__(self, user_dbname, sql_type, selectivity):
      self.sql_type = sql_type
      self.user_dbname = user_dbname
      self.selectivity = selectivity
      
      gt_dir = f'./benchmark/groundtruth/s{selectivity}'
      sql_dir = f'./benchmark/sql/s{selectivity}'
      mlir_dir = f'./benchmark/mlir/s{selectivity}'
      res_dir = f'./benchmark/results/s{self.selectivity}'
      
      self.gt_path = f'{gt_dir}/{sql_type}.csv'
      self.sql_file_path = f'{sql_dir}/{sql_type}.sql'
      
      self.ori_mlir_path = f'{mlir_dir}/{sql_type}_ori.mlir'
      self.opt_mlir_path = f'{mlir_dir}/{sql_type}_opt.mlir'
      
      self.res_path = f'{res_dir}/performance.csv'
      
      def create_dir(dir):
         if not os.path.exists(dir):
            os.makedirs(dir, mode=0o777)
      
      dirs = [gt_dir, mlir_dir, res_dir]
      for dir in dirs:
         create_dir(dir)
      
   def get_recall(self, result_df):
      # return 1
      gt_df = pd.read_csv(self.gt_path, sep='\t', index_col=0)

      def convert_float_to_str(df):
         for col in df.columns:
            if df[col].dtype == 'float64':
                  df[col] = df[col].apply(lambda x: format(x, '.5f'))
         return df
      result_df = convert_float_to_str(result_df).drop_duplicates()
      gt_df = convert_float_to_str(gt_df).drop_duplicates()
      

      def calculate_recall(result_df, gt_df):
         first_col_name = result_df.columns[0]

         result_grouped = result_df.groupby(first_col_name)
         gt_grouped = gt_df.groupby(first_col_name)
         total_recall = 0
         for key, result_group in result_grouped:
            if key in gt_grouped.groups:
                  gt_group = gt_grouped.get_group(key)
                  intersection_df = pd.merge(result_group, gt_group, how='inner')
                  recall = intersection_df.shape[0] / gt_group.shape[0]
                  total_recall += recall

         overall_recall = total_recall / gt_grouped.ngroups
         return overall_recall

      recall = calculate_recall(result_df, gt_df)
      return recall
   
   def append_to_csv(self, recall, execute_time, total_time):
      print(f"Recall = {recall:.4f}, ExecuteTime = {execute_time:.2f} ms, Total = {total_time:.2f} ms")
      
      df = pd.DataFrame({
         'DBName': self.user_dbname,
         'SQLType': self.sql_type,
         'Recall': [round(recall, 5)],
         'ExecuteTime': [round(execute_time, 5)],
         'Total': [round(total_time, 5)]
      })
      df.to_csv(self.res_path, mode='a', header=not pd.io.common.file_exists(self.res_path), index=True)    
      
   