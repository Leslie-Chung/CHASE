#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <arrow/pretty_print.h>
#include <arrow/table.h>

#include "execution/ResultProcessing.h"
#include "runtime/TableBuilder.h"
#include <functional>
#include <ostream>
#include <string>
#include <vector>
#include <arrow/io/api.h>


// #include <arrow/ipc/api.h>

namespace {
unsigned char hexval(unsigned char c) {
   if ('0' <= c && c <= '9')
      return c - '0';
   else if ('a' <= c && c <= 'f')
      return c - 'a' + 10;
   else if ('A' <= c && c <= 'F')
      return c - 'A' + 10;
   else
      abort();
}

class TableRetriever : public execution::ResultProcessor {
   std::shared_ptr<arrow::Table>& result;

   public:
   TableRetriever(std::shared_ptr<arrow::Table>& result) : result(result) {}
   void process(runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<runtime::ResultTable>(0);
      if (!resultTable) return;
      result = resultTable.value()->get();
   }
};

#if USE_FIXEDSIZEBINARY
std::string reverseHexStringBytes(const std::string& hexString) {
   std::string reversedHexString;
   for (int i = hexString.length() - 2; i >= 0; i -= 2) {
      reversedHexString += hexString.substr(i, 2);
   }
   return reversedHexString;
}

float hexStringToFloat(const std::string& hexString) {
   unsigned int intValue;
   std::stringstream ss;
   ss << std::hex << hexString;
   ss >> intValue;
   float floatValue;
   std::memcpy(&floatValue, &intValue, sizeof(float));
   return floatValue;
}
#endif

void printTable(const std::shared_ptr<arrow::Table>& table) {
   // Do not output anything for insert or copy statements
   if (table->columns().empty()) {
      std::cout << "Statement executed successfully." << std::endl;
      return;
   }
   auto now = std::chrono::system_clock::now();
   auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
   // std::stringstream ss;
   // ss << "/home/mr/lingo-db/results/" << now_ms << ".arrow";
   // std::string file(ss.str());

   // auto inputFile = arrow::io::FileOutputStream::Open(file).ValueOrDie();
   // auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, table->schema()).ValueOrDie();
   // if (!batchWriter->WriteTable(*table).ok() || !batchWriter->Close().ok() || !inputFile->Close().ok()) {
   //    std::cerr << "could not store table" << std::endl;
   //    exit(1);
   // }
   std::vector<std::string> columnReps;
   std::vector<size_t> positions;
   arrow::PrettyPrintOptions options;
   options.indent_size = 0;
   options.window = 100;
   std::cout << "|";
   std::string rowSep = "-";
   std::vector<bool> convertHex;
   std::vector<bool> convertVector;
   uint32_t vectorSize = 0;
   for (auto c : table->columns()) {
      std::cout << std::setw(30) << table->schema()->field(positions.size())->name() << "  |";
      bool isV;
#if USE_FIXEDSIZEBINARY
      isV = table->schema()->field(positions.size())->name().find("vec") != table->schema()->field(positions.size())->name().npos;
      if (isV) {
         convertVector.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
         vectorSize = table->schema()->field(positions.size())->type()->byte_width() / sizeof(float);
      } else {
         convertVector.push_back(false);
      }
#else
      convertVector.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_LIST); // support vector

#endif
      isV = false;
      convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY && !isV);

      rowSep += std::string(33, '-');
      std::stringstream sstr;
      if (convertVector.back()) {
         options.window = INT_MAX;
         options.container_window = INT_MAX;
      } else {
         options.window = 100;
      }

      arrow::PrettyPrint(*c.get(), options, &sstr); //NOLINT (clang-diagnostic-unused-result)
      columnReps.push_back(sstr.str());
#if !USE_FIXEDSIZEBINARY
      if (convertVector.back()) {
         std::string& ori = columnReps.back();
         for (size_t i = 0; i < ori.size(); ++i) {
            if (!isdigit(ori[i]) && ori[i] != '-') {
               continue;
            }
            size_t j = i - 1;
            while (ori[j] != ']') {
               if (ori[j] == '\n') {
                  ori[j] = '+';
               }
               ++j;
            }
            i = j + 1;
         }
      }
#endif
      // std::cout << "\n" <<  columnReps.back();
      positions.push_back(0);
   }

   std::cout << std::endl
             << rowSep << std::endl;
   bool cont = true;
   while (cont) { 
      cont = false;
      bool skipNL = false;
      for (size_t column = 0; column < columnReps.size(); column++) { 
         char lastHex = 0;
         bool first = true;
         std::stringstream out;

         while (positions[column] < columnReps[column].size()) { 
            cont = true;
            char curr = columnReps[column][positions[column]];
            char next = columnReps[column][positions[column] + 1];
            positions[column]++;
            if (first && (curr == '[' || curr == ']' || curr == ',')) {
               continue;
            }
            if (curr == ',' && next == '\n') {
               continue;
            }
            if (curr == '\n') {
               break;
            }

            if (convertHex[column]) {
               if (isxdigit(curr)) {
                  if (lastHex == 0) {
                     first = false;
                     lastHex = curr;
                  } else {
                     char converted = (hexval(lastHex) << 4 | hexval(curr));
                     out << converted;
                     lastHex = 0;
                  }
               } else {
                  first = false;
                  out << curr;
               }
            } else if (convertVector[column]) { // support vector
#if USE_FIXEDSIZEBINARY
               uint32_t itemPos = 0;
               out << '[';
               while (itemPos < vectorSize) {
                  std::string hexstr(&columnReps[column][positions[column] - 1], itemPos * 8, 8);
                  float item = hexStringToFloat(reverseHexStringBytes(hexstr));
                  out << std::fixed << std::setprecision(2) << item;
                  if (itemPos != vectorSize - 1)
                     out << ", ";
                  itemPos++;
               }
               out << ']';
               positions[column] += vectorSize * 8 - 1;
               first = false;
#else
               out << '[';
               while (curr != ']') {
                  if (curr != '\n' && curr != '+') {
                     out << curr;
                  }
                  if (curr == ',') {
                     out << ' ';
                  }
                  curr = columnReps[column][positions[column]++];
               }
               out << ']';

               first = false;
#endif
            } else {
               first = false;
               out << curr;
            }
         }
         if (first) {
            skipNL = true;
         } else {
            if (column == 0) {
               std::cout << "|";
            }
            std::cout << std::setw(30) << out.str() << "  |";
         }
      }
      if (!skipNL) {
         std::cout << "\n";
      }
   }
}

class TablePrinter : public execution::ResultProcessor {
   void process(runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<runtime::ResultTable>(0);
      if (!resultTable) return;
      auto table = resultTable.value()->get();
      printTable(table);
   }
};
class BatchedTablePrinter : public execution::ResultProcessor {
   void process(runtime::ExecutionContext* executionContext) override {
      for (size_t i = 0;; i++) {
         auto resultTable = executionContext->getResultOfType<runtime::ResultTable>(i);
         if (!resultTable) return;
         auto table = resultTable.value()->get();
         printTable(table);
      }
   }
};
} // namespace

std::unique_ptr<execution::ResultProcessor> execution::createTableRetriever(std::shared_ptr<arrow::Table>& result) {
   return std::make_unique<TableRetriever>(result);
}

std::unique_ptr<execution::ResultProcessor> execution::createTablePrinter() {
   return std::make_unique<TablePrinter>();
}

std::unique_ptr<execution::ResultProcessor> execution::createBatchedTablePrinter() {
   return std::make_unique<BatchedTablePrinter>();
}