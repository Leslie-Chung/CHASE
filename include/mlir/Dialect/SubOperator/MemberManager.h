#ifndef MLIR_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
#define MLIR_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
#include <algorithm>
#include <string>
#include <unordered_map>
namespace mlir::subop {
class MemberManager {
   std::unordered_map<std::string, size_t> counts;

   public:
   std::string getUniqueMember(std::string name) { // name == XXX$id
      auto sanitizedName = name.substr(0, name.find("$")); // XXX
      std::replace(sanitizedName.begin(), sanitizedName.end(), ' ', '_');  
      auto id = counts[sanitizedName]++;
      auto res = sanitizedName + "$" + std::to_string(id); // XXX$id+1
      return res;
   };
};
} // namespace mlir::subop

#endif //MLIR_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
