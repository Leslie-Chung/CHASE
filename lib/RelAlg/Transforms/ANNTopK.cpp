#include "llvm/ADT/TypeSwitch.h"
#include "mlir-support/eval.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsEnums.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stack"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>

namespace {
class ANNTopK : public mlir::PassWrapper<ANNTopK, mlir::OperationPass<mlir::func::FuncOp>> { // support vector
   virtual llvm::StringRef getArgument() const override { return "relalg-topk-with-ann"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ANNTopK)

   bool enableToTrans = false;

   bool containsANNIndex(mlir::relalg::BaseTableOp baseTableOp,
                         mlir::tuples::ColumnRefAttr& vectorColumn, std::string& tableName, std::string& indexName) { 
      bool res = false;
      std::unordered_map<std::string, bool> ANNKeyFound;
      for (auto indiceMetaData : baseTableOp.getMeta().getMeta()->getIndices()) {
         std::string col = indiceMetaData->columns[0];
         if (indiceMetaData->type == runtime::Index::Type::HNSW) {
            ANNKeyFound[col] = true;
         }
      }
      auto& colManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto* relevantColumn = &vectorColumn.getColumn();
      tableName = colManager.getName(relevantColumn).first;
      std::string columnName = colManager.getName(relevantColumn).second;

      if (baseTableOp.getCreatedColumns().contains(relevantColumn) && ANNKeyFound.contains(columnName)) { 
         res = true;
         indexName = columnName;
      }
      // res = true;
      return res;
   }

   void runOnOperation() override {
      auto baseTableOps = getOperation().getOps<mlir::relalg::BaseTableOp>();

      getOperation().walk([&](mlir::relalg::TopKOp topKOp) {
         mlir::relalg::SortSpecificationAttr firstSortSpec = topKOp.getSortspecs()[0].cast<mlir::relalg::SortSpecificationAttr>();

         if (firstSortSpec.getSortSpec() == mlir::relalg::SortSpec::desc) {
            return;
         }

         int sortBySize = topKOp.getSortspecs().size();
         mlir::relalg::MapOp mapOp = topKOp.getRel().getDefiningOp<mlir::relalg::MapOp>();
         while (--sortBySize) {
            mapOp = mapOp.getRel().getDefiningOp<mlir::relalg::MapOp>();
         }
         if (!mapOp) return;
         auto result = mapOp.getPredicate().front().getTerminator()->getOperand(0);
         auto distanceOp = result.getDefiningOp<mlir::relalg::DistanceOpInterface>();

         if (distanceOp) {
            mlir::Operation* op = result.getDefiningOp();
            int getColumnOpIndex = 0, constantOpIndex = 1;
            if (!op->getOperand(getColumnOpIndex).getDefiningOp<mlir::tuples::GetColumnOp>()) {
               getColumnOpIndex = 1;
               constantOpIndex = 0;
            }
            mlir::tuples::GetColumnOp getColumnOp = op->getOperand(getColumnOpIndex).getDefiningOp<mlir::tuples::GetColumnOp>();
            mlir::tuples::ColumnRefAttr vectorColumn = getColumnOp.getAttr();
            std::string tableName, indexName;
            for (auto baseTableOp : baseTableOps) {
               if (baseTableOp.getTableIdentifier() == vectorColumn.getName().getRootReference().getValue()) {
                  if (containsANNIndex(baseTableOp, vectorColumn, tableName, indexName)) {
                     break;
                  }
               }
            }
            
            if (!indexName.size()) {
               return;
            }

            enableToTrans = true;
            if (!mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(mapOp.getChildren()[0].getOperation())) { 
               mlir::relalg::SelectionOp currentSelection = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(mapOp.getChildren()[0].getOperation());
               std::stack<mlir::relalg::SelectionOp> upperSelections;
               upperSelections.push(currentSelection);
               while (currentSelection) { 
                  Operator child = currentSelection.getChildren()[0];
                  if (std::vector<mlir::Operation*>(child->getUsers().begin(), child->getUsers().end()).size() > 1) {
                     child = {};
                  }
                  currentSelection = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(child.getOperation());
                  if (currentSelection) { upperSelections.push(currentSelection); }
               }
               enableToTrans = false;
               if (upperSelections.size()) { 
                  mlir::relalg::SelectionOp firstSelectionOp = upperSelections.top();
                  mlir::relalg::BaseTableOp baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(firstSelectionOp.getRel().getDefiningOp());
                  if (baseTableOp) enableToTrans = true;
               } 
            } 

            if (!enableToTrans) return;
            mlir::db::ConstantOp constantOp = op->getOperand(constantOpIndex).getDefiningOp<mlir::db::ConstantOp>();

            mlir::Operation* opAfterMapOp = *mapOp->getUsers().begin(); 
            mlir::OpBuilder builder(opAfterMapOp);
            mlir::IntegerAttr distanceSpaceAttr = mlir::IntegerAttr::get(builder.getI8Type(), distanceOp.getDistanceSpace());
            mlir::DenseF32ArrayAttr query = constantOp.getValue().dyn_cast<mlir::DenseF32ArrayAttr>();


            auto annTopKOp =
               builder.create<mlir::relalg::ANNTopKOp>(mapOp.getLoc(), mapOp.getResult(), topKOp.getMaxRowsAttr(),
                                                       distanceSpaceAttr, query, vectorColumn,
                                                       mapOp.getComputedColsAttr()[0].cast<mlir::tuples::ColumnDefAttr>());
            // opAfterMapOp->setOperand(0, annTopKOp.getResult());
            opAfterMapOp->replaceUsesOfWith(mapOp.getResult(), annTopKOp.getResult());
            annTopKOp->setAttr("ann_index", mlir::StringAttr::get(annTopKOp.getContext(), indexName));
            annTopKOp->setAttr("table_identifier", mlir::StringAttr::get(annTopKOp.getContext(), tableName));
         }
      });
      // getOperation()->dump();
   }
};
} // end anonymous namespace

namespace mlir { namespace relalg {
std::unique_ptr<Pass> createANNTopKPass() { return std::make_unique<ANNTopK>(); }
} // end namespace relalg
} // end namespace mlir