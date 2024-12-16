#include "runtime/HNSWIndex.h"

#include "execution/Execution.h"
#include "runtime/helpers.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <tuple>

#include <arrow/api.h>
#include <arrow/array/array_primitive.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
namespace runtime {

HNSWIndex::HNSWIndex(Relation& r, std::vector<std::string> keyColumns, std::string index_name, std::string dbDir, size_t maxelements, size_t M, size_t efConstruction, size_t dim, uint8_t distanceSpace) 
   : Index(r, keyColumns), dbDir(dbDir), table(r.getTable()) {

   if (table && table->num_rows() != 0) {
      switch(distanceSpace) {
         case 1:
            distanceFunction = std::make_shared<hnswlib::L2Space>(dim);
            break;
         case 2:
            distanceFunction = std::make_shared<hnswlib::InnerProductSpace>(dim);
            break;
      }
      name = index_name;
      maxelements = table->num_rows();
      hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(distanceFunction.get(), maxelements, M, efConstruction);
      int i = 0;
      std::string& vectorColName = indexedColumns[0];
      auto vectorCol = table->GetColumnByName(vectorColName)->chunks();
      for (auto chunk : vectorCol) {
         std::shared_ptr<arrow::FixedSizeListArray> listArray = std::static_pointer_cast<arrow::FixedSizeListArray>(chunk);
         for (int j = 0; j < listArray->length(); j++) {
               std::shared_ptr<arrow::Array> list = listArray->value_slice(j);
               std::shared_ptr<arrow::FloatArray> floatArray = std::static_pointer_cast<arrow::FloatArray>(list);
               const float* constVector = floatArray->raw_values();
               float* vector = const_cast<float*>(constVector);
               hnsw->addPoint(vector, i++);
         }
      }
   }
}

void HNSWIndex::build(uint64_t startRow, std::shared_ptr<arrow::Table> tmpTable) {
   size_t numRows = tmpTable->num_rows();
   std::string& vectorColName = indexedColumns[0];
   auto vectorCol = tmpTable->GetColumnByName(vectorColName)->chunks();

   uint64_t dims = 0;
   for (auto chunk : vectorCol) {
      auto vecs = chunk->data()->child_data[0];
      dims += vecs->length;
   }
   uint64_t dim = dims / numRows;

   uint64_t i = 0;
   auto dataFile = dbDir + "/" + relation.getName() + "." + name + ".hnsw";
   uint8_t distanceSpace = relation.getMetaData()->getVectorDistanceSpace();
   switch(distanceSpace) {
      case 1:
         distanceFunction = std::make_shared<hnswlib::L2Space>(dim);
         break;
      case 2:
         distanceFunction = std::make_shared<hnswlib::InnerProductSpace>(dim);
         break;
   }
   size_t maxelement = relation.getMetaData()->getHnswIndexMaxElements();
   hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(distanceFunction.get(), dataFile, false, maxelement);

   for (auto chunk : vectorCol) {
      std::shared_ptr<arrow::FixedSizeListArray> listArray = std::static_pointer_cast<arrow::FixedSizeListArray>(chunk);
      for (int j = 0; j < listArray->length(); j++) {
            std::shared_ptr<arrow::Array> list = listArray->value_slice(j);
            std::shared_ptr<arrow::FloatArray> floatArray = std::static_pointer_cast<arrow::FloatArray>(list);
            const float* constVector = floatArray->raw_values();
            float* vector = const_cast<float*>(constVector);
            hnsw->addPoint(vector, i + startRow);
            i++;
      }
   }
}

void HNSWIndex::flush() {
   if (persist) {
      auto dataFile = dbDir + "/" + relation.getName() + "." + name + ".hnsw";
      hnsw->saveIndex(dataFile);
   }
}
void HNSWIndex::setPersist(bool value) {
   Index::setPersist(value);
   flush();
}

void HNSWIndex::ensureLoaded() {
   if (recordBatches.size()) return;
   auto dataFile = dbDir + "/" + relation.getName() + "." + name + ".hnsw";
   table = relation.getTable();
   if (std::filesystem::exists(dataFile)) {
      if (!distanceFunction.get()) { 
         distanceFunction = std::make_shared<hnswlib::L2Space>(0);
      }
      hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(distanceFunction.get(), dataFile);
   }
   // else {
   //    build();
   // }

   // load table data
   arrow::TableBatchReader reader(table);
   std::shared_ptr<arrow::RecordBatch> recordBatch;
   while (reader.ReadNext(&recordBatch).ok() && recordBatch) {
      recordBatches.push_back(recordBatch);
   }
}
void HNSWIndex::appendRows(std::shared_ptr<arrow::Table> toAppend) {
   std::vector<std::shared_ptr<arrow::RecordBatch>> newTableBatches;
   bool isEmpty = true;
   uint64_t startRow = 0;
   if (table && table->num_rows() != 0) {
      isEmpty = false;
      startRow = table->num_rows();
      newTableBatches.push_back(table->CombineChunksToBatch().ValueOrDie());
   }
   newTableBatches.push_back(toAppend->CombineChunksToBatch().ValueOrDie());
   table = arrow::Table::FromRecordBatches(newTableBatches).ValueOrDie();
   if (isEmpty) {
      build(0, table);
   } else {
      build(startRow, toAppend);
   }
   flush();
}


HNSWIndexIteration* HNSWIndexAccess::beginScan(const float* query, int k, float range) {
   return new HNSWIndexIteration(this, query, k, range);
}
void HNSWIndexIteration::close(runtime::HNSWIndexIteration* iteration) {
   delete iteration;
}

const int runtime::HNSWIndexIteration::queueThreshold = 50;
const int runtime::HNSWIndexIteration::distanceThreshold = 3;
bool HNSWIndexIteration::hasNext() { // condition
   // static int times = 0;
   int i = 0;

   while (true) {
      queryResult.SetHasResult(false);
      // times++;
      // std::cout << times << std::endl;
      access->hnswIndex->hnsw->searchKnnIterative(query, workSpace.get(), queryResult, isFirstResult);
      isFirstResult = false;
      if (!queryResult.HasResult()) {
         return false;
      }
      uint8_t res = checkFunc(this, i);
      if (res == CONTINUE) continue;
      return res;
   }
   return false;
}

inline uint8_t HNSWIndexIteration::checkRangeCondition(HNSWIndexIteration* self, int& i) {
   if (self->currentCategories == 0) {
      return FALSE;
   }
   float dis = self->queryResult.GetDistance();
   if (!self->inRange) {
      if (self->distanceQueue.size() < queueThreshold || self->distanceQueue.top() > dis)
      {
         if (self->distanceQueue.size() == queueThreshold) {
            self->distanceQueue.pop();
         }
         self->distanceQueue.push(dis); 
      } else {
         return FALSE;
      }
   }
   if (dis > self->range)
   {
      if (self->inRange) {
         i++;
         if (i >= distanceThreshold) {
            return FALSE;
         }
      }
      return CONTINUE;
   } else {
      self->inRange = true;
   }

   return TRUE;
}

void HNSWIndexIteration::doTop1() {
   std::priority_queue<float> tmpdistanceQueue;
   while (true) {
      queryResult.SetHasResult(false);
      access->hnswIndex->hnsw->searchKnnIterative(query, workSpace.get(), queryResult, isFirstResult);
      isFirstResult = false;
      if (!queryResult.HasResult()) {
         break;
      }
      float dis = queryResult.GetDistance();
      if (tmpdistanceQueue.size() == 86 && tmpdistanceQueue.top() < dis) {
         break;
      } else {
         if (tmpdistanceQueue.size() == 86) {
            tmpdistanceQueue.pop();
         }
         tmpdistanceQueue.push(dis);
      }
   }
   workSpace->Reset();
}

inline uint8_t HNSWIndexIteration::checkTopkCondition(HNSWIndexIteration* self, int& i) {
   if (self->hnswInorder && self->filteredK >= self->k) {
      return FALSE;
   }
   float dis = self->queryResult.GetDistance();
   if (!self->hnswInorder) {
      if (self->distanceQueue.size() == self->kRange && self->distanceQueue.top() < dis) {
         self->hnswInorder = true;
      } else {
         if (self->distanceQueue.size() == self->kRange) {
            self->distanceQueue.pop();
         }
         self->distanceQueue.push(dis);
      }
   }
   return TRUE;
}

void HNSWIndexIteration::setCategory(char* category) {
   auto it = categoryMap.find(category);
   auto& t = categoryMap[category];
   int filtered = ++std::get<2>(t);
   if (it == categoryMap.end()) { 
      if (currentCategories == -1) {
         currentCategories = 0;
      }
      currentCategories++;
      std::get<0>(t).push(queryResult.GetDistance());
      std::get<1>(t) = false;
      std::get<2>(t) = 1;
      std::get<3>(t) = false;
   } else {
      bool& categoryHnswInorder = std::get<1>(t);
      if (!categoryHnswInorder) {
         auto& categoryDistanceQueue = std::get<0>(t);
         if (categoryDistanceQueue.size() == kRange && categoryDistanceQueue.top() < queryResult.GetDistance()) {
            categoryHnswInorder = true;
         } else {
            if (categoryDistanceQueue.size() == kRange) {
               categoryDistanceQueue.pop();
            }
            categoryDistanceQueue.push(queryResult.GetDistance());
         }
      } else {
         bool& haveMinusCategories = std::get<3>(t);
         if (filtered >= k && !haveMinusCategories) {
            haveMinusCategories = true;
            currentCategories--;
         }
      }
   }
}

void HNSWIndexIteration::consumeRecordBatch(runtime::RecordBatchInfo* info) {
   uint64_t pos = 0;
   size_t tuple_pos = queryResult.GetLabel();
   for (size_t i = 0; i < access->recordBatchInfos.size(); ++i) {
      auto* targetInfo = access->recordBatchInfos[i];
      pos += targetInfo->numRows;
      if (tuple_pos < pos) {
         memcpy(info, targetInfo, access->infoSize);
         for (size_t j = 0; j != access->colIds.size(); ++j) {
            info->columnInfo[j].offset += (tuple_pos - (pos - targetInfo->numRows));
         }
         info->numRows = 1; 
         return;
      }
   }
}
HNSWIndexAccess::HNSWIndexAccess(std::shared_ptr<runtime::HNSWIndex> hnswIndex, std::vector<std::string> cols) : hnswIndex(hnswIndex) {
   // Find column ids for relevant columns
   for (auto columnToMap : cols) {
      auto columnNames = hnswIndex->table->ColumnNames();
      size_t columnId = 0;
      bool found = false;
      for (auto column : columnNames) {
         if (column == columnToMap) {
            colIds.push_back(columnId);
            found = true;
            break;
         }
         columnId++;
      }
      if (!found) throw std::runtime_error("column not found: " + columnToMap);
   }

   // Calculate size of RecordBatchInfo for relevant columns
   infoSize = colIds.size() * sizeof(ColumnInfo) + sizeof(RecordBatchInfo);

   // Prepare RecordBatchInfo for each record batch to facilitate computation for individual tuples at runtime
   for (auto& recordBatchPtr : hnswIndex->recordBatches) {
      RecordBatchInfo* recordBatchInfo = static_cast<RecordBatchInfo*>(malloc(infoSize));
      recordBatchInfo->numRows = recordBatchPtr->num_rows();
      for (size_t i = 0; i != colIds.size(); ++i) {
         auto colId = colIds[i];
         ColumnInfo& colInfo = recordBatchInfo->columnInfo[i];
         // Base offset for record batch, will need to add individual tuple offset in record batch
         colInfo.offset = recordBatchPtr->column_data(colId)->offset;
         if (recordBatchPtr->column_data(colId)->child_data.size()) {
            colInfo.offset += recordBatchPtr->column_data(colId)->child_data[0]->offset;
         }
         // Facilitates handling of null values
         colInfo.validMultiplier = recordBatchPtr->column_data(colId)->buffers[0] ? 1 : 0;
         // Compact representation of null values (inversed)
         colInfo.validBuffer = RecordBatchInfo::getBuffer(recordBatchPtr.get(), colId, 0);
         // Pointer to fixed size data for column
         colInfo.dataBuffer = RecordBatchInfo::getBuffer(recordBatchPtr.get(), colId, 1);
         // Pointer to variable length data for column
         colInfo.varLenBuffer = RecordBatchInfo::getBuffer(recordBatchPtr.get(), colId, 2);
      }
      recordBatchInfos.push_back(recordBatchInfo);
   }
}

std::shared_ptr<Index> Index::createHNSWIndex(runtime::IndexMetaData& metaData, runtime::Relation& relation, std::string dbDir) {
   auto res = std::make_shared<HNSWIndex>(relation, metaData.columns, dbDir);
   res->name = metaData.name;
   return res;
}
} // end namespace runtime