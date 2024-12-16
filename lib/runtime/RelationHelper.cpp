#include "runtime/RelationHelper.h"
#include "runtime/HNSW/space_ip.h"
#include "runtime/HNSWIndex.h"
#include "runtime/TableBuilder.h"

#include <cstdint>
#include <iostream>
#include <memory>

#include "json.h"

#include <arrow/csv/api.h>
#include <arrow/io/api.h>
namespace runtime {
void RelationHelper::createTable(runtime::ExecutionContext* context, runtime::VarLen32 name, runtime::VarLen32 meta) {
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   catalog->addTable(name.str(), runtime::TableMetaData::deserialize(meta.str()));
}
void RelationHelper::appendTableFromResult(runtime::VarLen32 tableName, runtime::ExecutionContext* context, size_t resultId) {
   {
      auto resultTable = context->getResultOfType<runtime::ResultTable>(resultId);
      if (!resultTable) {
         throw std::runtime_error("appending result table failed: no result table");
      }
      auto& session = context->getSession();
      auto catalog = session.getCatalog();
      if (auto relation = catalog->findRelation(tableName)) {
         relation->append(resultTable.value()->get());
         relation->setPersist(true);
      } else {
         throw std::runtime_error("appending result table failed: no such table");
      }
   }
}
void RelationHelper::copyFromIntoTable(runtime::ExecutionContext* context, runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape) {
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->findRelation(tableName)) {
      arrow::io::IOContext ioContext = arrow::io::default_io_context();
      auto inputFile = arrow::io::ReadableFile::Open(fileName.str()).ValueOrDie();
      std::shared_ptr<arrow::io::InputStream> input = inputFile;

      auto readOptions = arrow::csv::ReadOptions::Defaults();

      auto parseOptions = arrow::csv::ParseOptions::Defaults();
      parseOptions.delimiter = delimiter.str().front();
      if (escape.getLen() > 0) {
         parseOptions.escape_char = escape.str().front();
         parseOptions.escaping = true;
      }
      parseOptions.newlines_in_values = true;
      auto convertOptions = arrow::csv::ConvertOptions::Defaults();
      auto schema = relation->getArrowSchema();
      convertOptions.null_values.push_back("");
      convertOptions.strings_can_be_null = true;
      for (auto f : schema->fields()) {
         if (f->name().find("primaryKeyHashValue") != std::string::npos) continue;
         readOptions.column_names.push_back(f->name());
         convertOptions.column_types.insert({f->name(), f->type()});
      }

      // Instantiate TableReader from input stream and options
      auto maybeReader = arrow::csv::TableReader::Make(ioContext,
                                                       input,
                                                       readOptions,
                                                       parseOptions,
                                                       convertOptions);
      if (!maybeReader.ok()) {
         // Handle TableReader instantiation error...
      }
      std::shared_ptr<arrow::csv::TableReader> reader = *maybeReader;

      // Read table from CSV file
      auto maybeTable = reader->Read();
      if (!maybeTable.ok()) {
         // Handle CSV read error
         // (for example a CSV syntax error or failed type conversion)
      }
      std::shared_ptr<arrow::Table> table = *maybeTable;
      relation->append(table);
   } else {
      throw std::runtime_error("copy failed: no such table");
   }
}
void RelationHelper::setPersist(runtime::ExecutionContext* context, bool value) {
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   catalog->setPersist(value);
}

void RelationHelper::createHashIndex(runtime::ExecutionContext* context, runtime::VarLen32 relationName, runtime::VarLen32 indexName, runtime::VarLen32 keyColumnsDescription) {
   auto& session = context->getSession();
   std::shared_ptr<runtime::DBCatalog> catalog = std::dynamic_pointer_cast<runtime::DBCatalog>(session.getCatalog());
   if(auto relation = catalog->findRelation(relationName)) {
      auto metadata = relation->getMetaData();
      auto existingIndices = metadata->getIndices();
      for (auto& index : existingIndices) {
         if (index->type == Index::Type::HASH) {
            throw std::runtime_error("Hash index already exists");
         }
      }
      std::vector<std::string> columnStr;
      std::stringstream ss(keyColumnsDescription.str());
      std::string str;
      while (std::getline(ss, str, ',')) {
         columnStr.push_back(str);
      }

      auto hashIndex = new HashIndex(*relation, columnStr, catalog->getDBDirectory(), indexName.str());
      hashIndex->ensureLoaded();
      auto hashIndexMetaData = std::make_shared<IndexMetaData>();
      hashIndexMetaData->type = Index::Type::HASH;
      hashIndexMetaData->name = indexName.str();
      hashIndexMetaData->columns = columnStr;
      metadata->getIndices().push_back(hashIndexMetaData);
      relation->setIndex(indexName.str(), std::shared_ptr<Index>(hashIndex));
      relation->setPersist(true);
      hashIndex->setPersist(true);
   }
}
void RelationHelper::createHNSWIndex(runtime::ExecutionContext* context, size_t m, size_t efconstruction, size_t maxelements, runtime::VarLen32 distanceFunction, runtime::VarLen32 relationName) {
   auto& session = context->getSession();
   std::shared_ptr<runtime::DBCatalog> catalog = std::dynamic_pointer_cast<runtime::DBCatalog>(session.getCatalog());
   if (auto relation = catalog->findRelation(relationName)) {
      auto metadata = relation->getMetaData();
      auto existingIndices = metadata->getIndices();
      for (auto& index : existingIndices) {
         if (index->type == Index::Type::HNSW) {
            throw std::runtime_error("HNSW index already exists");
         }
      }
      uint8_t distanceSpace = 0;
      if (distanceFunction.str() == "vector_l2_ops") {
         distanceSpace = 1;
      } else if (distanceFunction.str() == "vector_innerproduct_ops") {
         distanceSpace = 2;
      }
      size_t dim = std::get<size_t>(metadata->getColumnMetaData(metadata->getVectorColumn()[0])->getColumnType().modifiers[0]);
      auto hnswIndex = new HNSWIndex(*relation, metadata->getVectorColumn(), metadata->getVectorColumn()[0], catalog->getDBDirectory(), maxelements, m, efconstruction, dim, distanceSpace);

      auto hnswIndexMetaData = std::make_shared<IndexMetaData>();
      hnswIndexMetaData->type = Index::Type::HNSW;
      hnswIndexMetaData->name = metadata->getVectorColumn()[0];
      hnswIndexMetaData->columns = metadata->getVectorColumn();
      metadata->getIndices().push_back(hnswIndexMetaData);
      relation->setIndex(metadata->getVectorColumn()[0], std::shared_ptr<Index>(hnswIndex));
      metadata->setVectorDistanceSpace(distanceSpace);
      metadata->setHnswIndexMaxElements(maxelements);
      relation->setPersist(true);
      hnswIndex->setPersist(true);
   } 
   else {
      throw std::runtime_error("no such table");
   }
}

HashIndexAccess* RelationHelper::getIndex(runtime::ExecutionContext* context, runtime::VarLen32 description) {
   auto json = nlohmann::json::parse(description.str());
   std::string relationName = json["relation"];
   std::string index = json["index"];
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->findRelation(relationName)) {
      auto* hashIndex = static_cast<HashIndex*>(relation->getIndex(index).get());
      std::vector<std::string> cols;
      for (auto m : json["mapping"].get<nlohmann::json::object_t>()) {
         cols.push_back(m.second.get<std::string>());
      }
      return new HashIndexAccess(*hashIndex, cols);
   } else {
      throw std::runtime_error("no such table");
   }
}

HNSWIndexAccess* RelationHelper::getHNSWIndex(runtime::ExecutionContext* context, runtime::VarLen32 description) {
   auto json = nlohmann::json::parse(description.str());
   std::string relationName = json["relation"];
   std::string index = json["index"];
   int dim = json["dim"];
   uint8_t distanceSpace = json["distance_space"];
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->findRelation(relationName)) {
      auto hnswIndex = std::dynamic_pointer_cast<HNSWIndex>(relation->getIndex(index));
      switch (distanceSpace) {
         case 1:
            hnswIndex->distanceFunction = std::make_shared<hnswlib::L2Space>(dim);
            break;
         case 2:
            hnswIndex->distanceFunction = std::make_shared<hnswlib::InnerProductSpace>(dim);
            break;
      }
      hnswIndex->hnsw->data_size_ = hnswIndex->distanceFunction->get_data_size();
      hnswIndex->hnsw->fstdistfunc_ = hnswIndex->distanceFunction->get_dist_func();
      hnswIndex->hnsw->dist_func_param_ = hnswIndex->distanceFunction->get_dist_func_param();
      std::vector<std::string> cols;
      for (auto m : json["mapping"].get<nlohmann::json::object_t>()) {
         cols.push_back(m.second.get<std::string>());
      }
      HNSWIndexAccess* hnswIndexAccess = new HNSWIndexAccess(hnswIndex, cols);
      context->registerState({hnswIndexAccess, [](void* ptr) { delete reinterpret_cast<HNSWIndexAccess*>(ptr); }});
      return hnswIndexAccess;
   } else {
      throw std::runtime_error("no such table");
   }
}
} // end namespace runtime