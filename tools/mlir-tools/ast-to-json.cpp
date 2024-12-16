#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "libpg_query/pg_query.h"
#include "runtime/Catalog.h"

void printMLIR(std::string sql, std::shared_ptr<runtime::Catalog> catalog) {
   PgQueryInternalParsetreeAndError result;

   pg_query_parse_init();
   result = pg_query_parse(sql.c_str());
   if (result.error) {
      std::cout << "Syntax Error at position " << result.error->cursorpos << ":" << result.error->message;
      return;
   }
   print_pg_parse_tree(result.tree);
   pg_query_free_parse_result(result);
}

int main(int argc, char** argv) {
   std::string filename = std::string(argv[1]);
   auto catalog = runtime::Catalog::createEmpty();
   if (argc >= 3) {
      std::string dbDir = std::string(argv[2]);
      catalog = runtime::DBCatalog::create(catalog, dbDir, false);
   }
   std::ifstream istream{filename};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   while (true) {
      std::stringstream query;
      std::string line;
      std::getline(buffer, line);
      while (true) {
         if (!buffer.good()) {
            if (buffer.eof()) {
               query << line << std::endl;
            }
            break;
         }
         query << line << std::endl;
         if (!line.empty() && line.find(';') == line.size() - 1) {
            break;
         }
         std::getline(buffer, line);
      }
      printMLIR(query.str(), catalog);
      if (buffer.eof()) {
         break;
      }
   }
   return 0;
}