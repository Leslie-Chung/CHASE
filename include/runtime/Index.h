#ifndef RUNTIME_INDEX_H
#define RUNTIME_INDEX_H
#include <string>
#include <vector>

#include <arrow/type_fwd.h>
namespace runtime {
struct IndexMetaData;
class Relation;
class Index {
   public:
   enum Type {
      HASH = 0,
      HNSW
   };

   protected:
   Type indexType;
   std::string name;
   std::vector<std::string> indexedColumns;
   Relation& relation;
   bool persist;

   public:
   Index(Relation& r, std::vector<std::string> indexedColumns) : indexedColumns(indexedColumns), relation(r) {}
   Index(Relation& r, std::vector<std::string> indexedColumns, std::string name) : indexedColumns(indexedColumns), relation(r), name(name) {}
   virtual void ensureLoaded() = 0;
   virtual void appendRows(std::shared_ptr<arrow::Table> table) = 0;
   virtual void setPersist(bool value) {
      persist = value;
   }
   static std::shared_ptr<Index> createHashIndex(IndexMetaData& metaData, Relation& relation, std::string dbDir);
   static std::shared_ptr<Index> createHNSWIndex(IndexMetaData& metaData, Relation& relation, std::string dbDir);
   virtual ~Index() {}
};

} //end namespace runtime
#endif //RUNTIME_INDEX_H
