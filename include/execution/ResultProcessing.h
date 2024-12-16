#ifndef EXECUTION_RESULTPROCESSING_H
#define EXECUTION_RESULTPROCESSING_H
#include "runtime/ExecutionContext.h"
#define USE_FIXEDSIZEBINARY 0 
namespace execution {
class ResultProcessor {
   public:
   virtual void process(runtime::ExecutionContext* executionContext) = 0;
   virtual ~ResultProcessor() {}
};
std::unique_ptr<ResultProcessor> createTablePrinter();
std::unique_ptr<ResultProcessor> createBatchedTablePrinter();
std::unique_ptr<ResultProcessor> createTableRetriever(std::shared_ptr<arrow::Table>& result);
} // namespace execution
#endif //EXECUTION_RESULTPROCESSING_H
