#ifndef RUNTIME_METADATA_H
#define RUNTIME_METADATA_H
#include <optional>
#include <unordered_map>
#include <variant>

#include "runtime/Index.h"

#include <arrow/record_batch.h>
namespace runtime {
struct ColumnType {
   std::string base;
   bool nullable;
   std::vector<std::variant<size_t, std::string>> modifiers;
};
class ColumnMetaData {
   std::optional<size_t> distinctValues;
   ColumnType columnType;

   public:
   const std::optional<size_t>& getDistinctValues() const;
   void setDistinctValues(const std::optional<size_t>& distinctValues);
   const ColumnType& getColumnType() const;
   void setColumnType(const ColumnType& columnType);
};
struct IndexMetaData {
   std::string name;
   Index::Type type;
   std::vector<std::string> columns;
};
class TableMetaData {
   bool present;
   size_t numRows;
   std::vector<std::string> primaryKey;
   std::unordered_map<std::string, std::shared_ptr<ColumnMetaData>> columns;
   std::vector<std::string> orderedColumns;
   std::shared_ptr<arrow::RecordBatch> sample;
   std::vector<std::shared_ptr<IndexMetaData>> indices;

   uint8_t vectorDistanceSpace = 0;
   size_t hnswIndexMaxElements = 0;

   public:
   TableMetaData() : present(false) {}
   size_t getNumRows() const {
      return numRows;
   }
   void setPresent() {
      present = true;
   }
   void setNumRows(size_t numRows) {
      TableMetaData::numRows = numRows;
   }
   void addColumn(std::string name, std::shared_ptr<ColumnMetaData> columnMetaData) {
      columns[name] = columnMetaData;
      orderedColumns.push_back(name);
   }
   void setPrimaryKey(const std::vector<std::string>& primaryKey) {
      TableMetaData::primaryKey = primaryKey;
   }
   const std::vector<std::string>& getPrimaryKey() const {
      return primaryKey;
   }
   const std::shared_ptr<ColumnMetaData> getColumnMetaData(const std::string& name) const {
      return columns.at(name);
   }
   void setVectorDistanceSpace(const uint8_t vectorDistanceSpace) {
      TableMetaData::vectorDistanceSpace = vectorDistanceSpace;
   }
   void setHnswIndexMaxElements(const size_t max_elements) {
      TableMetaData::hnswIndexMaxElements = max_elements;
   }
   const std::vector<std::string>& getVectorColumn() const;
   const uint8_t getVectorDistanceSpace() const {
      return vectorDistanceSpace;
   }
   const size_t getHnswIndexMaxElements() const {
      return hnswIndexMaxElements;
   }
   const std::shared_ptr<arrow::RecordBatch>& getSample() const {
      return sample;
   }
   std::vector<std::shared_ptr<IndexMetaData>>& getIndices() {
      return indices;
   }
   const std::vector<std::string>& getOrderedColumns() const;
   static std::shared_ptr<TableMetaData> deserialize(std::string);
   std::string serialize(bool serializeSample = true) const;
   static std::shared_ptr<TableMetaData> create(const std::string& json, const std::string& name, std::shared_ptr<arrow::RecordBatch> sample);
   bool isPresent() const;
};
} // end namespace runtime

#endif // RUNTIME_METADATA_H
