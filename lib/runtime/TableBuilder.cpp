#include "runtime/TableBuilder.h"
#include "arrow/array/array_binary.h"
#include "arrow/array/builder_base.h"
#include "arrow/array/builder_nested.h"
#include "arrow/type_fwd.h"
#include "runtime/helpers.h"
#include <iostream>
#include <string>
#include <vector>

#include "utility/Tracer.h"
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/util/decimal.h>

#define USE_FIXEDSIZEBINARY 0
namespace {
static utility::Tracer::Event tableBuilderMerge("TableBuilder", "merge");
} // end namespace
class TableBuilder {
   static constexpr size_t maxBatchSize = 100000;
   std::shared_ptr<arrow::Schema> schema;
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   std::unique_ptr<arrow::RecordBatchBuilder> batchBuilder;
   size_t currentBatchSize = 0;
   size_t currColumn = 0;

   static std::shared_ptr<arrow::DataType> createType(std::string name, uint32_t p1, uint32_t p2) {
      if (name == "int") {
         switch (p1) {
            case 8: return arrow::int8();
            case 16: return arrow::int16();
            case 32: return arrow::int32();
            case 64: return arrow::int64();
         }
      } else if (name == "uint") {
         switch (p1) {
            case 8: return arrow::uint8();
            case 16: return arrow::uint16();
            case 32: return arrow::uint32();
            case 64: return arrow::uint64();
         }
      } else if (name == "float") {
         switch (p1) {
            case 16: return arrow::float16();
            case 32: return arrow::float32();
            case 64: return arrow::float64();
         }
      } else if (name == "string") {
         return arrow::utf8();
      } else if (name == "fixed_sized") {
         return arrow::fixed_size_binary(p1);
      } else if (name == "date") {
         return p1 == 32 ? arrow::date32() : arrow::date64();
      } else if (name == "interval_months") {
         return arrow::month_interval();
      } else if (name == "interval_daytime") {
         return arrow::day_time_interval();
      } else if (name == "timestamp") {
         return arrow::timestamp(static_cast<arrow::TimeUnit::type>(p1));
      } else if (name == "decimal") {
         return arrow::decimal(p1, p2);
      } else if (name == "bool") {
         return arrow::boolean();
      } else if (name == "vector") { // support vector
         #if USE_FIXEDSIZEBINARY
         return arrow::fixed_size_binary(p1 * sizeof(float));
         #else
         return arrow::fixed_size_list(arrow::float32(), p1);
         #endif
      }
      throw std::runtime_error("unknown type");
   }
   static std::shared_ptr<arrow::Schema> parseSchema(std::string str) {
      if (str.empty()) {
         return std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{});
      }
      std::vector<std::shared_ptr<arrow::Field>> fields;

      str.erase(std::remove_if(str.begin(), str.end(), [](char c) { return c == ' '; }), str.end());
      auto parseEntry = [&fields](std::string token) { 
         size_t colonPos = token.find(":");
         if (colonPos == std::string::npos) throw std::runtime_error("expected ':'");
         std::string colName = token.substr(0, colonPos);
         std::string typeDescr = token.substr(colonPos + 1);
         size_t lParamPos = typeDescr.find("[");
         std::string p1 = "0";
         std::string p2 = "0";
         std::string typeName = typeDescr;
         if (lParamPos != std::string ::npos) {
            typeName = typeDescr.substr(0, lParamPos);
            assert(typeDescr.ends_with(']'));
            std::string paramString = typeDescr.substr(lParamPos + 1, typeDescr.size() - lParamPos - 2);
            size_t commaPos = paramString.find(",");
            if (commaPos == std::string::npos) {
               p1 = paramString;
            } else {
               p1 = paramString.substr(0, commaPos);
               p2 = paramString.substr(commaPos + 1);
            }
         }
         fields.push_back(std::make_shared<arrow::Field>(colName, createType(typeName, std::stoi(p1), std::stoi(p2))));
      };
      size_t pos = 0;
      std::string token;
      while ((pos = str.find(";")) != std::string::npos) {
         token = str.substr(0, pos);
         str.erase(0, pos + 1);
         parseEntry(token);
      }
      parseEntry(str);
      return std::make_shared<arrow::Schema>(fields);
   }
   std::shared_ptr<arrow::Schema> lowerSchema(std::shared_ptr<arrow::Schema> schema) {
      std::vector<std::shared_ptr<arrow::Field>> fields;
      for (auto f : schema->fields()) {
         auto t = GetPhysicalType(f->type()); //
         fields.push_back(std::make_shared<arrow::Field>(f->name(), t));
      }
      auto lowered = std::make_shared<arrow::Schema>(fields);
      // std::cout<<"lowered:"<<lowered->ToString()<<std::endl;
      return lowered;
   }
   TableBuilder(std::shared_ptr<arrow::Schema> schema) : schema(schema) {
      batchBuilder=arrow::RecordBatchBuilder::Make(lowerSchema(schema), arrow::default_memory_pool()).ValueOrDie();
   }
   std::shared_ptr<arrow::RecordBatch> convertBatch(std::shared_ptr<arrow::RecordBatch> recordBatch) {
      std::vector<std::shared_ptr<arrow::ArrayData>> columnData;
      for (int i = 0; i < recordBatch->num_columns(); i++) {
         #if USE_FIXEDSIZEBINARY
         columnData.push_back(arrow::ArrayData::Make(schema->field(i)->type(), recordBatch->column_data(i)->length, recordBatch->column_data(i)->buffers, recordBatch->column_data(i)->null_count, recordBatch->column_data(i)->offset));
         #else
         if (schema->field(i)->type()->id() != arrow::Type::FIXED_SIZE_LIST) {
            columnData.push_back(arrow::ArrayData::Make(schema->field(i)->type(), recordBatch->column_data(i)->length, recordBatch->column_data(i)->buffers, recordBatch->column_data(i)->null_count, recordBatch->column_data(i)->offset));
         } else { // support vector
            auto pData = recordBatch->column_data(i);
         //    auto childData = pData->child_data[0];
         //    auto parent = arrow::ArrayData::Make(schema->field(i)->type(), pData->length, pData->buffers, pData->null_count, pData->offset);  
         //    parent->child_data.push_back(arrow::ArrayData::Make(arrow::float32(), childData->length, childData->buffers, childData->null_count, childData->offset));
            auto parent = arrow::ArrayData::Make(schema->field(i)->type(), pData->length, pData->buffers, pData->child_data, pData->dictionary, pData->null_count, pData->offset); 
            columnData.push_back(parent);
         }
         #endif
      }
      return arrow::RecordBatch::Make(schema, recordBatch->num_rows(), columnData);
   }
   void flushBatch() {
      if (currentBatchSize > 0) {
         std::shared_ptr<arrow::RecordBatch> recordBatch;
         recordBatch=batchBuilder->Flush(true).ValueOrDie(); //NOLINT (clang-diagnostic-unused-result)
         currentBatchSize = 0;
         batches.push_back(convertBatch(recordBatch));
      }
   }
   template <typename T>
   T* getBuilder() {
      auto ptr = batchBuilder->GetFieldAs<T>(currColumn++);
      assert(ptr != nullptr);
      return ptr;
   }
   void handleStatus(arrow::Status status) {
      if (!status.ok()) {
         throw std::runtime_error(status.ToString());
      }
   }

   public:
   static TableBuilder* create(runtime::VarLen32 schemaDescription);
   std::shared_ptr<arrow::Table> build();

   void addBool(bool isValid, bool value);
   void addInt8(bool isValid, int8_t);
   void addInt16(bool isValid, int16_t);
   void addInt32(bool isValid, int32_t);
   void addInt64(bool isValid, int64_t);
   void addFloat32(bool isValid, float);
   void addFloat64(bool isValid, double);
   void addDecimal(bool isValid, __int128);
   void addFixedSized(bool isValid, int64_t);
   void addBinary(bool isValid, runtime::VarLen32);
   void addVector(bool isValid, float *); // support vector
   void nextRow();
   void merge(TableBuilder* other) {
      other->flushBatch();
      batches.insert(batches.end(), other->batches.begin(), other->batches.end());
      other->batches.clear();
   }
};

#define EXPORT extern "C" __attribute__((visibility("default")))

TableBuilder* TableBuilder::create(runtime::VarLen32 schemaDescription) {
   return new TableBuilder(parseSchema(schemaDescription.str()));
}

std::shared_ptr<arrow::Table> TableBuilder::build() {
   flushBatch();
   std::shared_ptr<arrow::Table> table;
   // auto currChunk = batches[0];
   // int colId = 0;
   // std::vector<std::shared_ptr<arrow::ArrayData>> &childArray = currChunk.get()->column_data(colId)->child_data;
   // std::cerr << childArray.size() << std::endl;
   // int64_t* data0 = (int64_t*)childArray[0]->buffers[1]->address();
   // std::cerr << data0[0] << std::endl;
   // std::cerr << data0[2] << std::endl;

   auto st = arrow::Table::FromRecordBatches(schema, batches).Value(&table);
   if (!st.ok()) {
      throw std::runtime_error("could not create table:" + st.ToString());
   }
   return table;
}
void TableBuilder::addBool(bool isValid, bool value) {
   auto* typedBuilder = getBuilder<arrow::BooleanBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(value));
   }
}

#define TABLE_BUILDER_ADD_PRIMITIVE(name, type)                                                \
   void TableBuilder::add##name(bool isValid, arrow::type ::c_type val) {                      \
      auto* typedBuilder = getBuilder<arrow::NumericBuilder<arrow::type>>();                   \
      if (!isValid) {                                                                          \
         handleStatus(typedBuilder->AppendNull()); /*NOLINT (clang-diagnostic-unused-result)*/ \
      } else {                                                                                 \
         handleStatus(typedBuilder->Append(val)); /*NOLINT (clang-diagnostic-unused-result)*/  \
      }                                                                                        \
   }

TABLE_BUILDER_ADD_PRIMITIVE(Int8, Int8Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int16, Int16Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int32, Int32Type)
TABLE_BUILDER_ADD_PRIMITIVE(Int64, Int64Type)
TABLE_BUILDER_ADD_PRIMITIVE(Float32, FloatType)
TABLE_BUILDER_ADD_PRIMITIVE(Float64, DoubleType)

void TableBuilder::addDecimal(bool isValid, __int128 value) {
   auto* typedBuilder = getBuilder<arrow::Decimal128Builder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(value >> 64, value));
      handleStatus(typedBuilder->Append(decimalrep));
   }
}
void TableBuilder::addBinary(bool isValid, runtime::VarLen32 string) {
   auto* typedBuilder = getBuilder<arrow::BinaryBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      std::string str = (string).str();
      handleStatus(typedBuilder->Append(string.getPtr(), string.getLen()));
   }
}
void TableBuilder::addFixedSized(bool isValid, int64_t val) {
   auto* typedBuilder = getBuilder<arrow::FixedSizeBinaryBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append(reinterpret_cast<char*>(&val)));
   }
}

void TableBuilder::addVector(bool isValid, float* vec) { // support vector
   #if USE_FIXEDSIZEBINARY
   auto* typedBuilder = getBuilder<arrow::FixedSizeBinaryBuilder>();
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append((char *)vec));
   }
   #else
   auto* typedBuilder = getBuilder<arrow::FixedSizeListBuilder>();
   const auto& fixed_size_list_type = arrow::internal::checked_pointer_cast<arrow::FixedSizeListType>(typedBuilder->type());
   if (!isValid) {
      handleStatus(typedBuilder->AppendNull());
   } else {
      handleStatus(typedBuilder->Append());
      arrow::FloatBuilder *value_builder = (arrow::FloatBuilder *)typedBuilder->value_builder();
      handleStatus(value_builder->AppendValues(vec, vec + fixed_size_list_type->list_size()));
   }
   #endif
}


void TableBuilder::nextRow() {
   assert(currColumn == static_cast<size_t>(schema->num_fields()));
   currColumn = 0;
   if (currentBatchSize >= maxBatchSize) {
      flushBatch();
   }
   currentBatchSize++;
}

#define RESULT_TABLE_FORWARD(name, type)                     \
   void runtime::ResultTable::name(bool isValid, type val) { \
      builder->name(isValid, val);                           \
   }

RESULT_TABLE_FORWARD(addBool, bool);
RESULT_TABLE_FORWARD(addInt8, int8_t);
RESULT_TABLE_FORWARD(addInt16, int16_t);
RESULT_TABLE_FORWARD(addInt32, int32_t);
RESULT_TABLE_FORWARD(addInt64, int64_t);
RESULT_TABLE_FORWARD(addFloat32, float);
RESULT_TABLE_FORWARD(addFloat64, double);
RESULT_TABLE_FORWARD(addDecimal, __int128);
RESULT_TABLE_FORWARD(addBinary, runtime::VarLen32);
RESULT_TABLE_FORWARD(addFixedSized, int64_t);
RESULT_TABLE_FORWARD(addVector, float*); // support vector
void runtime::ResultTable::nextRow() {
   builder->nextRow();
}
runtime::ResultTable* runtime::ResultTable::create(runtime::ExecutionContext* executionContext, runtime::VarLen32 schemaDescription) {
   ResultTable* resultTable = new ResultTable;
   resultTable->builder = TableBuilder::create(schemaDescription);
   executionContext->registerState({resultTable, [](void* ptr) { delete reinterpret_cast<ResultTable*>(ptr); }});
   return resultTable;
}
std::shared_ptr<arrow::Table> runtime::ResultTable::get() {
   if (resultTable) {
      return resultTable;
   } else {
      resultTable = builder->build();
   }
   return resultTable;
}

runtime::ResultTable* runtime::ResultTable::merge(runtime::ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(tableBuilderMerge);
   ResultTable* first = nullptr;
   for (auto* ptr : threadLocal->getTls()) {
      auto* current = reinterpret_cast<ResultTable*>(ptr);
      if (!first) {
         first = current;
      } else {
         first->builder->merge(current->builder);
      }
   }
   trace.stop();
   return first;
}