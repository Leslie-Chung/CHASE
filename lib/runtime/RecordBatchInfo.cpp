#include "runtime/RecordBatchInfo.h"
#include <arrow/array.h>
#include <arrow/record_batch.h>

#define USE_FIXEDSIZEBINARY 0
uint8_t* runtime::RecordBatchInfo::getBuffer(arrow::RecordBatch* batch, size_t columnId, size_t bufferId)  {
   static uint8_t alternative = 0b11111111;
   auto data = batch->column_data(columnId);
   #if !USE_FIXEDSIZEBINARY
   if(data->child_data.size() && bufferId != 0) { // support vector
      data = data->child_data[0];
   }
   #endif
   if (data->buffers.size() > bufferId && data->buffers[bufferId]) {
      auto* buffer = data->buffers[bufferId].get();
      return (uint8_t*) buffer->address();
   } else {
      return &alternative; //always return valid pointer to at least one byte filled with ones
   }
}