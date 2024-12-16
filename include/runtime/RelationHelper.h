#ifndef RUNTIME_RELATIONHELPER_H
#define RUNTIME_RELATIONHELPER_H

#include "ExecutionContext.h"
#include "HashIndex.h"
#include "helpers.h"
#include "HNSWIndex.h"
#include <cstddef>
#include <cstdint>
namespace runtime {
class RelationHelper {
   public:
   static void createTable(runtime::ExecutionContext* context, runtime::VarLen32 name, runtime::VarLen32 meta);
   static void appendTableFromResult(runtime::VarLen32 tableName, runtime::ExecutionContext* context, size_t resultId);
   static void copyFromIntoTable(runtime::ExecutionContext* context, runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape);
   static void setPersist(runtime::ExecutionContext* context, bool value);
   static HashIndexAccess* getIndex(runtime::ExecutionContext* context, runtime::VarLen32 description);
   static HNSWIndexAccess* getHNSWIndex(runtime::ExecutionContext* context, runtime::VarLen32 description);
   static void createHashIndex(runtime::ExecutionContext* context, runtime::VarLen32 relationName, runtime::VarLen32 indexName, runtime::VarLen32 keyColumnsDescription);
   static void createHNSWIndex(runtime::ExecutionContext* context, size_t m, size_t efconstruction, size_t maxelements, runtime::VarLen32 distanceFunction, runtime::VarLen32 relationName);
};
} // end namespace runtime

#endif //RUNTIME_RELATIONHELPER_H
