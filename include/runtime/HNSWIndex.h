#ifndef RUNTIME_HNSWINDEX_H
#define RUNTIME_HNSWINDEX_H
#include "HNSW/hnswalg.h"
#include "HNSW/query_result.h"
#include "HNSW/workspace.h"
#include "Index.h"
#include "runtime/RecordBatchInfo.h"
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <arrow/type_fwd.h>
namespace runtime {
class HNSWIndexIteration;
class HNSWIndexAccess;
class HNSWIndex : public Index {
   std::shared_ptr<arrow::Table> table;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches; 
   std::string dbDir;

   void build(uint64_t startRow, std::shared_ptr<arrow::Table> toAppend);

   public:
   HNSWIndex(Relation& r, std::vector<std::string> keyColumns, std::string dbDir) : Index(r, keyColumns), dbDir(dbDir) {}
   HNSWIndex(Relation& r, std::vector<std::string> keyColumns, std::string name, std::string dbDir, size_t maxelements, size_t M, size_t efConstruction, size_t dim, uint8_t distanceSpace);
   std::shared_ptr<hnswlib::SpaceInterface<float>> distanceFunction;
   std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
   void flush();
   void ensureLoaded() override;
   void appendRows(std::shared_ptr<arrow::Table> table) override;
   void setPersist(bool value) override;
   ~HNSWIndex() {
      distanceFunction = nullptr;
      hnsw = nullptr;
   }
   friend class HNSWIndexAccess;
   friend class HNSWIndexIteration;
};
class HNSWIndexAccess { 
   std::shared_ptr<HNSWIndex> hnswIndex;
   std::vector<size_t> colIds;
   std::vector<RecordBatchInfo*> recordBatchInfos; 
   size_t infoSize;

   public:
   HNSWIndexAccess(std::shared_ptr<runtime::HNSWIndex> hnswIndex, std::vector<std::string> cols);
   ~HNSWIndexAccess() {
      hnswIndex = nullptr;
      for (auto& recordBatchPtr : recordBatchInfos) {
         free(recordBatchPtr);
      }
   }
   HNSWIndexIteration* beginScan(const float* query, int k, float range);
   friend class HNSWIndexIteration;
};
class HNSWIndexIteration { 
   HNSWIndexAccess* access;
   const float* query;
   hnswlib::QueryResult<float> queryResult;
   std::shared_ptr<hnswlib::WorkSpace<float>> workSpace; // for hnsw search
   std::priority_queue<float> distanceQueue; // for hasNext condition
   bool isFirstResult = true;
   bool hnswInorder;
   bool inRange;
   bool pad;
   int kRange;
   float range;
   int filteredK;
   int k;

   // int times;
   static const int queueThreshold;
   static const int distanceThreshold;

   std::unordered_map<char*, std::tuple<std::priority_queue<float>, bool, int, bool>> categoryMap; // queue inorder filtered haveMinusCategories
   int currentCategories = -1;

   enum ReturnType {
      FALSE,
      TRUE,
      CONTINUE
   };
   uint8_t (*checkFunc)(HNSWIndexIteration*, int&);
   static inline uint8_t checkRangeCondition(HNSWIndexIteration* self, int& i);
   static inline uint8_t checkTopkCondition(HNSWIndexIteration* self, int &i);

   public:
   HNSWIndexIteration(HNSWIndexAccess* access, const float* query, int k, float range) : access(access), query(query), range(range), k(k)
   , isFirstResult(true), currentCategories(-1), kRange(86), inRange(false), hnswInorder(false), filteredK(0){
      // times = 0;
      workSpace = access->hnswIndex->hnsw->getFreeWorkSpace();
      if (k == -1 || (range != 86 && k != -1)) { // range || category/category join
         // hasRangeFilter = true;
         checkFunc = checkRangeCondition;
         if (range != 86 && k != -1) { // category/category join
            kRange = 43;
         }
      } else {
         checkFunc = checkTopkCondition;
         // kRange = range;
         kRange = 48;
      }
   }
   ~HNSWIndexIteration() {
      access->hnswIndex->hnsw->searchIndexIterativeEnd(workSpace);
      access = nullptr;
      // std::cout << times << std::endl;
   }
   bool hasNext();
   void setCategory(char*);

   void consumeRecordBatch(RecordBatchInfo*);
   float getCurrentDistance() {
      return queryResult.GetDistance(); 
   }
   bool gethnswInorder() {
      return hnswInorder;
   }
   void addFilteredK() {
      ++filteredK;
   }

   static void close(HNSWIndexIteration* iteration);
   void doTop1();
};

} //end namespace runtime
#endif //RUNTIME_HNSWINDEX_H
