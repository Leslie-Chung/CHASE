#include "llvm/ADT/TypeSwitch.h"
#include "mlir-support/eval.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBOpsEnums.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
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
class ANNRangeSelection : public mlir::PassWrapper<ANNRangeSelection, mlir::OperationPass<mlir::func::FuncOp>> { // support vector
   virtual llvm::StringRef getArgument() const override { return "relalg-range-selection-with-ann"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ANNRangeSelection)

   bool hasFindRangeSelection = false;

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


   void rangeSelectionWithANN(mlir::relalg::SelectionOp selectionOp, mlir::tuples::ColumnRefAttr& vectorColumn,
                              mlir::db::ConstantOp rangeConstantOp, mlir::db::ConstantOp queryConstantOp,
                              uint8_t distanceSpace, std::vector<mlir::Operation*>& toErase) {
      std::stack<mlir::relalg::SelectionOp> upperSelections;
      upperSelections.push(selectionOp);
      mlir::relalg::SelectionOp currentSelection = selectionOp;
      while (currentSelection) { 
         Operator child = currentSelection.getChildren()[0];
         if (std::vector<mlir::Operation*>(child->getUsers().begin(), child->getUsers().end()).size() > 1) {
            child = {};
         }
         currentSelection = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(child.getOperation());
         if (currentSelection) { upperSelections.push(currentSelection); }
      }

      mlir::relalg::SelectionOp firstSelectionOp = upperSelections.top();
      mlir::relalg::BaseTableOp baseTableOp =
         mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(firstSelectionOp.getRel().getDefiningOp());

      if (baseTableOp) {
         hasFindRangeSelection = true;
         std::string tableName, indexName;
         if (!containsANNIndex(baseTableOp, vectorColumn, tableName, indexName)) return;

         currentSelection = selectionOp;
         std::vector<mlir::relalg::SelectionOp> selections;
         while (!upperSelections.empty()) {
            selections.push_back(upperSelections.top());
            upperSelections.pop();
         }
         selections.pop_back(); 

         while (currentSelection) {
            Operator child = mlir::dyn_cast_or_null<Operator>(*(currentSelection->getUsers().begin()));
            if (!child ||
                std::vector<mlir::Operation*>(child->getUsers().begin(), child->getUsers().end()).size() > 1) {
               child = {};
            }
            currentSelection = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(child.getOperation());
            if (currentSelection) { selections.push_back(currentSelection); }
         }

         if (selections.size() && firstSelectionOp != selectionOp) {
            selectionOp.getResult().replaceAllUsesWith(
               selectionOp.getRel().getDefiningOp()->getResult(0)); 
            selectionOp->moveBefore(firstSelectionOp);
            selectionOp.setOperand(firstSelectionOp.getRel());
            firstSelectionOp.setOperand(selectionOp.getResult());
         }

         mlir::OpBuilder builder(selectionOp);
         mlir::IntegerAttr distanceSpaceAttr = mlir::IntegerAttr::get(builder.getI8Type(), distanceSpace);

         mlir::DenseF32ArrayAttr query = queryConstantOp.getValue().dyn_cast<mlir::DenseF32ArrayAttr>();
         mlir::FloatAttr range;
         mlir::Attribute tmp = rangeConstantOp.getValue();
         auto* cxt = builder.getContext();
         if (auto integerAttr = tmp.dyn_cast_or_null<mlir::IntegerAttr>()) {
            range = mlir::FloatAttr::get(mlir::Float32Type::get(cxt), integerAttr.getInt());
         } else if (auto floatAttr = tmp.dyn_cast_or_null<mlir::FloatAttr>()) {
            range = mlir::FloatAttr::get(mlir::Float32Type::get(cxt), floatAttr.getValueAsDouble());
         } else {
            auto stringAttr = tmp.dyn_cast_or_null<mlir::StringAttr>();
            range = mlir::FloatAttr::get(mlir::Float32Type::get(cxt), std::stof(stringAttr.str()));
         }

         auto& attrManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
         auto annAttr = attrManager.createDef(attrManager.getUniqueScope("ann"), "distance");
         annAttr.getColumn().type = (mlir::Type) builder.getF32Type();

         auto annRangeSelectionOp =
            builder.create<mlir::relalg::ANNRangeSelectionOp>(selectionOp.getLoc(), selectionOp.getRel(), range,
                                                              distanceSpaceAttr, query, vectorColumn,
                                                              annAttr);

         selectionOp.replaceAllUsesWith(annRangeSelectionOp.asRelation());
         annRangeSelectionOp->setAttr("ann_index", mlir::StringAttr::get(annRangeSelectionOp.getContext(), indexName));
         annRangeSelectionOp->setAttr("table_identifier", mlir::StringAttr::get(annRangeSelectionOp.getContext(), tableName));
         toErase.push_back(selectionOp.getOperation());
      }
   }

   void runOnOperation() override {
      auto ops = getOperation().getOps<mlir::relalg::SelectionOp>();
      std::vector<mlir::Operation*> toErase;

      for (mlir::relalg::SelectionOp selectionOp : ops) {
         if (hasFindRangeSelection) { break; }
         int used = 0;
         mlir::tuples::ColumnRefAttr vectorColumn; 
         selectionOp.walk([&](mlir::tuples::GetColumnOp getColumnOp) {
            ++used;
            if (getBaseType(getColumnOp.getAttr().getColumn().type).isa<mlir::db::DenseVectorType>()) {
               vectorColumn = getColumnOp.getAttr();
            }
         });
         if (used == 1 && vectorColumn) { 
            mlir::db::ConstantOp queryConstantOp = nullptr;
            selectionOp.walk([&](mlir::Operation* op) {
               if (hasFindRangeSelection) return;
               ::llvm::TypeSwitch<mlir::Operation*, void>(op)
                  .Case<mlir::db::ConstantOp>([&](mlir::db::ConstantOp constantOp) {
                     auto resType = constantOp.getResult().getType();
                     if (resType.isa<mlir::db::DenseVectorType>()) {
                        queryConstantOp = constantOp; // create query
                     }
                  })
                  .Case<mlir::db::CmpOp>([&](mlir::db::CmpOp cmpOp) {
                     mlir::Value constantOperand = cmpOp.getRight(), distanceOperand = cmpOp.getLeft();
                     // if (cmpOp.isLessPred(true) || cmpOp.isLessPred(false)) {
                     //    constantOperand = cmpOp.getRight(); // l2distance < constant range
                     //    distanceOperand = cmpOp.getLeft();

                     // } else if (cmpOp.isGreaterPred(true) || cmpOp.isGreaterPred(false)) {
                     //    constantOperand = cmpOp.getLeft();
                     //    distanceOperand = cmpOp.getRight();
                     // }
                     if (cmpOp.isGreaterPred(true) || cmpOp.isGreaterPred(false)) {
                        constantOperand = cmpOp.getLeft();
                        distanceOperand = cmpOp.getRight();
                     }
                     auto rangeConstantOp = constantOperand.getDefiningOp<mlir::db::ConstantOp>();
                     auto distanceOp = distanceOperand.getDefiningOp<mlir::relalg::DistanceOpInterface>();
                     if (rangeConstantOp && distanceOp &&
                         queryConstantOp) { 
                        rangeSelectionWithANN(selectionOp, vectorColumn, rangeConstantOp, queryConstantOp, distanceOp.getDistanceSpace(),
                                              toErase);
                     }
                  });
            });
         }
      }
      for (auto* op : toErase) { op->erase(); }
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createANNRangeSelectionPass() { return std::make_unique<ANNRangeSelection>(); }
} // end namespace relalg
} // end namespace mlir