#include <fstream>
#include <iostream>
#include <string>

#include "execution/Execution.h"
#include "execution/Timing.h"
#include "mlir-support/eval.h"

#include <stdlib.h>

int main(int argc, char** argv) {
   if (argc <= 2) {
      std::cerr << "USAGE: run-sql *.sql database eagerLoading" << std::endl;
      return 1;
   }
   bool eagerLoading = false;
   if (argc >= 4 && !strcmp(argv[3], "eagerLoading")) {
      eagerLoading = true;
   }
   std::string inputFileName = std::string(argv[1]);
   std::string directory = std::string(argv[2]);
   std::cout << "Loading Database from: " << directory << '\n';
   auto session = runtime::Session::createSession(directory,eagerLoading);

   support::eval::init();
   execution::ExecutionMode runMode = execution::getExecutionMode();
   auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, true);
   if (const char* numRuns = std::getenv("QUERY_RUNS")) {
      queryExecutionConfig->executionBackend->setNumRepetitions(std::atoi(numRuns));
      std::cout << "using " << queryExecutionConfig->executionBackend->getNumRepetitions() << " runs" << std::endl;
   }
   unsetenv("PERF_BUILDID_DIR");
   queryExecutionConfig->timingProcessor = std::make_unique<execution::TimingPrinter>(inputFileName);
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   executer->execute();
   return 0;
}
