#ifndef EXECUTION_FRONTEND_H
#define EXECUTION_FRONTEND_H
#include "Error.h"
#include "runtime/Catalog.h"
namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir
namespace execution {
class Frontend {
   protected:
   runtime::Catalog* catalog;
   Error error;

   std::unordered_map<std::string, double> timing;

   public:
   runtime::Catalog* getCatalog() const {
      return catalog;
   }
   void setCatalog(runtime::Catalog* catalog) {
      Frontend::catalog = catalog;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   virtual void loadFromFile(std::string fileName) = 0;
   virtual void loadFromString(std::string data) = 0;
   virtual bool isParallelismAllowed() { return true; }
   virtual mlir::ModuleOp* getModule() = 0;
   virtual ~Frontend() {}
};
std::unique_ptr<Frontend> createMLIRFrontend();
std::unique_ptr<Frontend> createSQLFrontend();
void initializeContext(mlir::MLIRContext& context);

} //namespace execution

#endif //EXECUTION_FRONTEND_H
