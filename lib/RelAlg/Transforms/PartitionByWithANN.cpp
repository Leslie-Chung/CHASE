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
#include "mlir/Dialect/TupleStream/Column.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
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
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
class PartitionByWithANN : public mlir::PassWrapper<PartitionByWithANN, mlir::OperationPass<mlir::func::FuncOp>> { // support vector
   virtual llvm::StringRef getArgument() const override { return "relalg-partitionby-with-ann"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionByWithANN)

   bool isBaseRelationWithSelects(Operator op, std::vector<mlir::Operation*>& path) {
      // Saves operations until base relation is reached on stack for easy access
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op.getOperation())
         .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp baseTableOp) {
            path.push_back(baseTableOp.getOperation());
            return true;
         })
         .Case<mlir::relalg::ANNRangeSelectionOp>([&](mlir::relalg::ANNRangeSelectionOp selectionOp) {
            path.push_back(selectionOp.getOperation());
            for (auto& child : selectionOp.getChildren()) {
               if (!isBaseRelationWithSelects(mlir::cast<Operator>(child.getOperation()), path)) return false;
            }
            return true;
         })
         .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp selectionOp) {
            path.push_back(selectionOp.getOperation());
            for (auto& child : selectionOp.getChildren()) {
               if (!isBaseRelationWithSelects(mlir::cast<Operator>(child.getOperation()), path)) return false;
            }
            return true;
         })
         .Default([&](auto&) {
            return false;
         });
   }

   bool isJoinWithSelects(Operator op, std::vector<mlir::Operation*>& path) {
      // Saves operations until base relation is reached on stack for easy access
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op.getOperation())
         .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp baseTableOp) {
            path.push_back(baseTableOp.getOperation());
            return true;
         })
         .Case<mlir::relalg::InnerJoinOp, mlir::relalg::CrossProductOp, mlir::relalg::SelectionOp>([&](Operator op) {
            path.push_back(op.getOperation());
            for (auto& child : op.getChildren()) {
               if (!isJoinWithSelects(mlir::cast<Operator>(child.getOperation()), path)) return false;
            }
            return true;
         })
         .Default([&](auto&) {
            return false;
         });
   }

   bool containsExactlyPrimaryKey(mlir::MLIRContext* ctxt, mlir::relalg::BaseTableOp baseTableOp, mlir::ArrayAttr partitionBy) {
      auto& colManager = ctxt->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      // Initialize map to verify presence of all primary key attributes
      std::unordered_map<std::string, bool> primaryKeyFound;
      for (auto primaryKeyAttribute : baseTableOp.getMeta().getMeta()->getPrimaryKey()) {
         primaryKeyFound[primaryKeyAttribute] = false;
      }

      // Verify all cmp operations
      bool res = true;
      for (auto column : partitionBy) {
         const mlir::tuples::Column* relevantColumn = column.cast<mlir::tuples::ColumnRefAttr>().getColumnPtr().get();
         if (baseTableOp.getCreatedColumns().contains(relevantColumn)) {
            std::string tableName = colManager.getName(relevantColumn).first;
            std::string columnName = colManager.getName(relevantColumn).second;
            if (!primaryKeyFound.contains(columnName))
               res = false;
            else
               primaryKeyFound[columnName] = true;
         } else {
            res = false;
         }
      }
      // Check if all primary key attributes were found
      for (auto primaryKeyAttribute : primaryKeyFound) {
         res &= primaryKeyAttribute.second;
      }
      return res;
   }

   bool containsANNIndex(mlir::relalg::BaseTableOp baseTableOp,
                         mlir::tuples::ColumnRefAttr vectorColumn, std::string& indexName) { 
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
      std::string columnName = colManager.getName(relevantColumn).second;

      if (baseTableOp.getCreatedColumns().contains(relevantColumn) && ANNKeyFound.contains(columnName)) { 
         res = true;
         indexName = columnName;
      }
      // res = true;
      return res;
   }

   bool columnsFromOneTable(mlir::MLIRContext* ctxt, mlir::relalg::BaseTableOp baseTableOp, mlir::ArrayAttr partitionBy, std::vector<mlir::relalg::MapOp>& mapOps) {
      std::unordered_set<const mlir::tuples::Column*> mappedCols;
      std::unordered_set<const mlir::tuples::Column*> oriCols;
      for (auto mapOp : mapOps) {
         mappedCols.insert(mapOp.getComputedCols()[0].cast<mlir::tuples::ColumnDefAttr>().getColumnPtr().get());
         oriCols.insert(*(mapOp.getUsedColumns().begin()));
      }
      for (auto attr : partitionBy) {
         const mlir::tuples::Column* col = attr.cast<mlir::tuples::ColumnRefAttr>().getColumnPtr().get();
         if (!mappedCols.contains(col)) { 
            oriCols.insert(col);
         }
      }

      bool res = true;
      for (auto* col : oriCols) {
         if (!baseTableOp.getCreatedColumns().contains(col)) {
            res = false;
            break;
         }
      }
      return res;
   }

   bool columnsFromTwoTableWithPK(mlir::MLIRContext* ctxt, mlir::relalg::BaseTableOp baseTableOp, mlir::relalg::BaseTableOp otherBaseTableOp, mlir::ArrayAttr partitionBy, std::vector<mlir::relalg::MapOp>& mapOps) {
      std::unordered_set<const mlir::tuples::Column*> mappedCols;
      std::unordered_set<const mlir::tuples::Column*> oriCols;
      for (auto mapOp : mapOps) {
         mappedCols.insert(mapOp.getComputedCols()[0].cast<mlir::tuples::ColumnDefAttr>().getColumnPtr().get());
         oriCols.insert(*(mapOp.getUsedColumns().begin()));
      }
      for (auto attr : partitionBy) {
         const mlir::tuples::Column* col = attr.cast<mlir::tuples::ColumnRefAttr>().getColumnPtr().get();
         if (!mappedCols.contains(col)) { 
            oriCols.insert(col);
         }
      }

      std::vector<mlir::Attribute> pks;
      auto& colManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      for (auto* col : oriCols) {
         if (!baseTableOp.getCreatedColumns().contains(col)) {
            pks.push_back(colManager.createRef(col));
         }
      }
      if (containsExactlyPrimaryKey(ctxt, otherBaseTableOp, mlir::ArrayAttr::get(ctxt, pks))) {
         return true;
      }
      return false;
   }

   void prepareForKNNJoin(BinaryOperator binOp, mlir::relalg::DistanceOpInterface distanceOp) {
      mlir::OpBuilder builder(binOp->getContext());
      auto dim = builder.getI32IntegerAttr(getBaseType(distanceOp.getLeft().getType()).cast<mlir::db::DenseVectorType>().getSize());
      auto distanceSpace = builder.getI8IntegerAttr(distanceOp.getDistanceSpace());
      auto range = builder.getF32FloatAttr(86);
      auto k = builder.getI32IntegerAttr(-1);

      auto leftCol = distanceOp.getLeft().getDefiningOp<mlir::tuples::GetColumnOp>().getAttr();
      auto rightCol = distanceOp.getRight().getDefiningOp<mlir::tuples::GetColumnOp>().getAttr();
      std::vector<mlir::Attribute> leftANN{leftCol, dim, distanceSpace, range, k}, rightANN;
      rightANN.assign(leftANN.begin(), leftANN.end());
      rightANN[0] = rightCol;

      binOp->setAttr("leftHash", builder.getArrayAttr(leftANN)); // colname dim distancespace range_constant k_constant
      binOp->setAttr("nullsEqual", builder.getArrayAttr({}));
      binOp->setAttr("rightHash", builder.getArrayAttr(rightANN));
   }

   mlir::IntegerAttr findKInSelection(mlir::relalg::WindowOp windowOp) {
      auto selectionOps = getOperation().getOps<mlir::relalg::SelectionOp>();
      auto rankCol = windowOp.getComputedColsAttr()[0].cast<mlir::tuples::ColumnDefAttr>();
      for (auto selectionOp : selectionOps) {
         int used = 0;
         bool find = false;
         selectionOp.getPredicate().walk([&](mlir::tuples::GetColumnOp getColumnOp) {
            ++used;
            if (getColumnOp.getAttr().getName() == rankCol.getName()) {
               find = true;
            }
         });
         if (used == 1 && find) {
            auto& predicate = selectionOp.getPredicate();
            auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(predicate.front().getTerminator());
            if (auto cmpOp = returnOp.getResults()[0].getDefiningOp<mlir::db::CmpOp>()) {
               mlir::Value constantOperand = {}; // cmpOp.isLessPred(true) || cmpOp.isLessPred(false)
               if (cmpOp.isLessPred(true) || cmpOp.isLessPred(false)) {
                  constantOperand = cmpOp.getRight(); // rank < k
               } else if (cmpOp.isGreaterPred(true) || cmpOp.isGreaterPred(false)) {
                  constantOperand = cmpOp.getLeft(); // k > rank
               }
               if (constantOperand) {
                  if (auto kConstantOp = constantOperand.getDefiningOp<mlir::db::ConstantOp>()) {
                     return kConstantOp.getValue().cast<mlir::IntegerAttr>();
                  }
               }
            }
         }
      }
      return {};
   }

   void runOnOperation() override {
      std::vector<mlir::Operation*> toErase;

      auto baseTables = getOperation().getOps<mlir::relalg::BaseTableOp>();

      getOperation().walk([&](mlir::relalg::WindowOp windowOp) {
         if (windowOp.getOrderByAttr().empty()) return;
         mlir::relalg::SortSpecificationAttr firstSortSpec = windowOp.getOrderByAttr()[0].cast<mlir::relalg::SortSpecificationAttr>();
         if (firstSortSpec.getSortSpec() != mlir::relalg::SortSpec::asc ||
             (int64_t) windowOp.getFrom() != ~INT64_MAX || windowOp.getTo() != 0) {
            return;
         }

         bool hasRank = false;
         windowOp.getAggrFunc().walk([&](mlir::relalg::RankOp) {
            hasRank = true;
         });
         if (!hasRank) return;

         int mappedPartition = 0;
         for (auto attr : windowOp.getPartitionByAttr()) {
            bool findMappedCol = true;
            auto partitionAttr = attr.cast<mlir::tuples::ColumnRefAttr>();
            for (auto baseTableOp : baseTables) {
               if (baseTableOp.getCreatedColumns().contains(partitionAttr.getColumnPtr().get())) {
                  findMappedCol = false;
                  break;
               }
            }
            if (findMappedCol) mappedPartition++;
         }
         std::vector<mlir::relalg::MapOp> mapOps;
         mlir::relalg::MapOp mapOp = windowOp.getRel().getDefiningOp<mlir::relalg::MapOp>(); 
         while (mappedPartition--) {
            mapOps.push_back(mapOp);
            mapOp = mapOp.getRel().getDefiningOp<mlir::relalg::MapOp>();
         }

         if (!mapOp) return;
         auto result = mapOp.getPredicate().front().getTerminator()->getOperand(0);
         auto distanceOp = result.getDefiningOp<mlir::relalg::DistanceOpInterface>();

         if (distanceOp) {
            int sortBySize = windowOp.getOrderBy().size();
            auto tmpMapOp = mapOp;
            while (--sortBySize) {
               tmpMapOp = tmpMapOp.getRel().getDefiningOp<mlir::relalg::MapOp>();
            }

            Operator op = tmpMapOp.getRel().getDefiningOp<Operator>();

            if (!op) return;
            std::vector<mlir::Operation*> path;
            if (!isBaseRelationWithSelects(op, path)) { 
               path.clear();
               if (isJoinWithSelects(op, path)) { 
                  int joinPos, leftBaseTablePos;
                  joinPos = leftBaseTablePos = 0;
                  for (int i = 0; i < path.size(); ++i) {
                     if (auto joinOp = mlir::dyn_cast_or_null<BinaryOperator>(path[i])) {
                        if (!joinPos)
                           joinPos = i;
                        else
                           return; 
                     } else if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(path[i])) {
                        if (!leftBaseTablePos) leftBaseTablePos = i;
                     }
                  }
                  auto binOp = mlir::dyn_cast_or_null<BinaryOperator>(path[joinPos]);
                  auto leftBaseTable = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(path[leftBaseTablePos]);
                  auto rightBaseTable = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(path[path.size() - 1]);
                  mlir::tuples::GetColumnOp leftVecOp = distanceOp.getLeft().getDefiningOp<mlir::tuples::GetColumnOp>();
                  mlir::tuples::GetColumnOp rightVecOp = distanceOp.getRight().getDefiningOp<mlir::tuples::GetColumnOp>();
                  if (!leftVecOp || !rightVecOp) return;
                  {
                     mlir::tuples::Column* leftVecRelevantColumn = leftVecOp.getAttr().cast<mlir::tuples::ColumnRefAttr>().getColumnPtr().get();
                     bool leftIsLeft = leftBaseTable.getCreatedColumns().contains(leftVecRelevantColumn);
                     if (!leftIsLeft) {
                        mlir::tuples::GetColumnOp tmp = leftVecOp;
                        leftVecOp = rightVecOp;
                        rightVecOp = tmp;
                     }
                     distanceOp->setOperands(mlir::ValueRange{leftVecOp, rightVecOp});
                  }
                  mlir::tuples::Column* leftVecRelevantColumn = leftVecOp.getAttr().cast<mlir::tuples::ColumnRefAttr>().getColumnPtr().get();
                  bool leftIsLeft = leftBaseTable.getCreatedColumns().contains(leftVecRelevantColumn);
                  mlir::tuples::Column* rightVecRelevantColumn = rightVecOp.getAttr().cast<mlir::tuples::ColumnRefAttr>().getColumnPtr().get();
                  bool rightIsRight = rightBaseTable.getCreatedColumns().contains(rightVecRelevantColumn);
                  if (!leftIsLeft || !rightIsRight) return;

                  if (binOp->hasAttr("useANNIndexNestedLoop")) {
                     bool leftTableCategoryJoin = columnsFromTwoTableWithPK(windowOp.getContext(), leftBaseTable, rightBaseTable, windowOp.getPartitionBy(), mapOps);
                     bool rightTableCategoryJoin = columnsFromTwoTableWithPK(windowOp.getContext(), rightBaseTable, leftBaseTable, windowOp.getPartitionBy(), mapOps);
                     if (leftTableCategoryJoin || rightTableCategoryJoin) {
                        std::string leftIndexName, rightIndexName;
                        bool leftCanUseANNIndex = containsANNIndex(leftBaseTable, leftVecOp.getAttr().cast<mlir::tuples::ColumnRefAttr>(), leftIndexName);
                        bool rightCanUseANNIndex = containsANNIndex(rightBaseTable, rightVecOp.getAttr().cast<mlir::tuples::ColumnRefAttr>(), rightIndexName);
                        if ((leftTableCategoryJoin && !leftCanUseANNIndex) || (rightTableCategoryJoin && !rightCanUseANNIndex)) { 
                           return;
                        }
                        auto leftHashInJoin = binOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
                        auto rightHashInJoin = binOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
                        auto leftVecInJoin = leftHashInJoin[2].cast<mlir::tuples::ColumnRefAttr>();
                        auto rightVecInJoin = rightHashInJoin[0].cast<mlir::tuples::ColumnRefAttr>();
                        auto distanceSpaceInJoin = rightHashInJoin[2].cast<mlir::IntegerAttr>();
                        if (leftVecInJoin.getColumnPtr().get() != leftVecRelevantColumn || rightVecInJoin.getColumnPtr().get() != rightVecRelevantColumn ||
                            distanceSpaceInJoin.getInt() != distanceOp.getDistanceSpace()) {
                           return;
                        }

                        auto k = findKInSelection(windowOp);
                        if (!k) {
                           return;
                        }

                        std::stack<mlir::Operation*> leftPath, rightPath;
                        for (int i = 0; i < joinPos; ++i) {
                           leftPath.push(path[i]);
                        }
                        leftPath.push(path[leftBaseTablePos]);
                        for (int i = leftBaseTablePos + 1; i < path.size(); ++i) {
                           rightPath.push(path[i]);
                        }
                        auto left = mlir::cast<Operator>(binOp.leftChild());
                        auto right = mlir::cast<Operator>(binOp.rightChild());

                        bool reversed = false;
                        mlir::OpBuilder builder(binOp);

                        {
                           auto tmp = binOp->getAttr("leftHash").cast<mlir::ArrayAttr>();
                           std::vector<mlir::Attribute> leftANN(tmp.begin(), tmp.end());
                           leftANN.erase(leftANN.begin(), leftANN.begin() + 2);
                           binOp->setAttr("leftHash", builder.getArrayAttr(leftANN));
                        }

                        if (rightTableCategoryJoin && rightCanUseANNIndex) {
                           reversed = true;
                           mlir::Operation* lastMoved = leftBaseTable.getOperation();
                           mlir::Operation* firstMoved = nullptr;

                           leftPath.pop();

                           // Move selections on left side before join
                           while (!leftPath.empty()) {
                              if (!firstMoved) firstMoved = leftPath.top();
                              leftPath.top()->moveAfter(lastMoved);
                              leftPath.top()->setOperands(mlir::ValueRange{lastMoved->getResult(0)});
                              lastMoved = leftPath.top();
                              leftPath.pop();
                           }

                           // If selections were moved, replace usages of join with last moved selection
                           if (firstMoved) {
                              binOp->replaceAllUsesWith(mlir::ValueRange{lastMoved->getResults()});
                              firstMoved->setOperands(binOp->getResults());
                           }

                           std::swap(left, right);
                           std::swap(leftPath, rightPath);
                           std::swap(leftIndexName, rightIndexName);
                           mlir::Attribute tmp = binOp->getAttr("rightHash");
                           binOp->setAttr("rightHash", binOp->getAttr("leftHash"));
                           binOp->setAttr("leftHash", tmp);
                        }
                        auto leftBaseTable = mlir::cast<mlir::relalg::BaseTableOp>(leftPath.top());
                        leftPath.pop();
                        // update binOp
                        binOp->setOperands(mlir::ValueRange{leftBaseTable, binOp->getOperand(!reversed)});
                        mlir::Operation* lastMoved = binOp.getOperation();
                        mlir::Operation* firstMoved = nullptr;

                        // Move selections on left side after join
                        while (!leftPath.empty()) {
                           if (!firstMoved) firstMoved = leftPath.top();
                           leftPath.top()->moveAfter(lastMoved);
                           leftPath.top()->setOperands(mlir::ValueRange{lastMoved->getResult(0)});
                           lastMoved = leftPath.top();
                           leftPath.pop();
                        }

                        // If selections were moved, replace usages of join with last moved selection
                        if (firstMoved) {
                           binOp->replaceAllUsesWith(mlir::ValueRange{lastMoved->getResults()});
                           firstMoved->setOperands(binOp->getResults());
                        }

                        // Add name of table to leftHash annotation
                        std::vector<mlir::Attribute> leftANN; // tablename indexname | colname dim distancespace range_constant k_constant computedCol
                        leftANN.push_back(leftBaseTable.getTableIdentifierAttr());
                        leftANN.push_back(mlir::StringAttr::get(binOp->getContext(), leftIndexName));
                        for (auto attr : binOp->getAttr("leftHash").dyn_cast_or_null<mlir::ArrayAttr>()) {
                           leftANN.push_back(attr);
                        }
                        leftANN.push_back(mapOp.getComputedCols()[0]); // computed column
                        // binOp->setAttr("leftHash", mlir::ArrayAttr::get(binOp->getContext(), leftANN));

                        {
                           // tablename indexname | colname dim distancespace range_constant k_constant computedCol
                           leftANN[6] = k; // set k
                           binOp->setAttr("leftHash", mlir::ArrayAttr::get(binOp->getContext(), leftANN));
                        }
                        {
                           // colname dim distancespace range_constant k_constant
                           mlir::ArrayAttr rightANN = binOp->getAttr("rightHash").dyn_cast<mlir::ArrayAttr>();
                           std::vector<mlir::Attribute> tmp(rightANN.begin(), rightANN.end());
                           tmp[4] = k; // set k
                           binOp->setAttr("rightHash", mlir::ArrayAttr::get(binOp->getContext(), tmp));
                        }
                        binOp->setAttr("useANNIndexNestedLoop", mlir::UnitAttr::get(binOp->getContext()));

                        windowOp->setAttr("FilteredK", mlir::UnitAttr::get(windowOp->getContext()));
                        windowOp->setAttr("needPartition", mlir::UnitAttr::get(windowOp->getContext()));
                        leftBaseTable->setAttr("virtual", mlir::UnitAttr::get(leftBaseTable->getContext()));

                        mapOp->replaceAllUsesWith(mlir::ValueRange{mapOp.getRel()});
                        toErase.push_back(mapOp.getOperation());
                     }

                  } else {
                     bool leftTableKnn = containsExactlyPrimaryKey(windowOp.getContext(), leftBaseTable, windowOp.getPartitionBy());
                     bool rightTableKnn = containsExactlyPrimaryKey(windowOp.getContext(), rightBaseTable, windowOp.getPartitionBy());
                     if (leftTableKnn || rightTableKnn) {
                        std::string leftIndexName, rightIndexName;
                        bool leftCanUseANNIndex = containsANNIndex(leftBaseTable, leftVecOp.getAttr().cast<mlir::tuples::ColumnRefAttr>(), leftIndexName);
                        bool rightCanUseANNIndex = containsANNIndex(rightBaseTable, rightVecOp.getAttr().cast<mlir::tuples::ColumnRefAttr>(), rightIndexName);
                        if (leftTableKnn) {
                           if (!rightCanUseANNIndex) return;
                        } else if (rightTableKnn) {
                           if (!leftCanUseANNIndex) {
                              return;
                           }
                        }
                        bool reversed = false;
                        auto left = mlir::cast<Operator>(binOp.leftChild());
                        auto right = mlir::cast<Operator>(binOp.rightChild());
                        std::stack<mlir::Operation*> leftPath, rightPath;
                        for (int i = joinPos + 1; i <= leftBaseTablePos; ++i) {
                           leftPath.push(path[i]);
                        }
                        for (int i = leftBaseTablePos + 1; i < path.size(); ++i) {
                           rightPath.push(path[i]);
                        }
                        auto k = findKInSelection(windowOp);
                        if (!k) {
                           return;
                        }
                        prepareForKNNJoin(binOp, distanceOp);

                        if (leftTableKnn && rightCanUseANNIndex) { 
                           reversed = true;
                           std::swap(left, right);
                           std::swap(leftPath, rightPath);
                           std::swap(leftIndexName, rightIndexName);
                           mlir::Attribute tmp = binOp->getAttr("rightHash");
                           binOp->setAttr("rightHash", binOp->getAttr("leftHash"));
                           binOp->setAttr("leftHash", tmp);
                        }
                        auto leftBaseTable = mlir::cast<mlir::relalg::BaseTableOp>(leftPath.top());
                        leftPath.pop();
                        // update binOp
                        binOp->setOperands(mlir::ValueRange{leftBaseTable, binOp->getOperand(!reversed)});
                        mlir::Operation* lastMoved = binOp.getOperation();
                        mlir::Operation* firstMoved = nullptr;

                        mlir::OpBuilder builder(binOp);

                        // Move selections on left side after join
                        while (!leftPath.empty()) {
                           if (!firstMoved) firstMoved = leftPath.top();
                           leftPath.top()->moveAfter(lastMoved);
                           leftPath.top()->setOperands(mlir::ValueRange{lastMoved->getResult(0)});
                           lastMoved = leftPath.top();
                           leftPath.pop();
                        }

                        // If selections were moved, replace usages of join with last moved selection
                        if (firstMoved) {
                           binOp->replaceAllUsesWith(mlir::ValueRange{lastMoved->getResults()});
                           firstMoved->setOperands(binOp->getResults());
                        }

                        // Add name of table to leftHash annotation
                        std::vector<mlir::Attribute> leftANN; // tablename indexname | colname dim distancespace range_constant k_constant computedCol
                        leftANN.push_back(leftBaseTable.getTableIdentifierAttr());
                        leftANN.push_back(mlir::StringAttr::get(binOp->getContext(), leftIndexName));
                        for (auto attr : binOp->getAttr("leftHash").dyn_cast_or_null<mlir::ArrayAttr>()) {
                           leftANN.push_back(attr);
                        }
                        leftANN.push_back(mapOp.getComputedCols()[0]); // computed column
                        // binOp->setAttr("leftHash", mlir::ArrayAttr::get(binOp->getContext(), leftANN));

                        binOp->setAttr("useANNIndexNestedLoop", mlir::UnitAttr::get(binOp->getContext()));

                        windowOp->setAttr("FilteredK", mlir::UnitAttr::get(windowOp->getContext())); 
                        leftBaseTable->setAttr("virtual", mlir::UnitAttr::get(leftBaseTable->getContext()));

                        {
                           // tablename indexname | colname dim distancespace range_constant k_constant computedCol
                           leftANN[6] = k; // set k
                           binOp->setAttr("leftHash", mlir::ArrayAttr::get(binOp->getContext(), leftANN));
                        }
                        {
                           // colname dim distancespace range_constant k_constant
                           mlir::ArrayAttr rightANN = binOp->getAttr("rightHash").dyn_cast<mlir::ArrayAttr>();
                           std::vector<mlir::Attribute> tmp(rightANN.begin(), rightANN.end());
                           tmp[4] = k; // set k
                           binOp->setAttr("rightHash", mlir::ArrayAttr::get(binOp->getContext(), tmp));
                        }
                        mapOp->replaceAllUsesWith(mlir::ValueRange{mapOp.getRel()});
                        toErase.push_back(mapOp.getOperation());
                     }
                  }
               }
            } else { 
               size_t pathSize = path.size();
               if (pathSize > 1) {
                  auto annRangeSelectionOp = mlir::dyn_cast_or_null<mlir::relalg::ANNRangeSelectionOp>(path[pathSize - 2]);
                  if (!annRangeSelectionOp) return;

                  auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(path[pathSize - 1]);
                  bool isPrimary = containsExactlyPrimaryKey(windowOp.getContext(), baseTableOp, windowOp.getPartitionBy());
                  if (isPrimary) return;

                  mlir::db::ConstantOp windowConstantOp = distanceOp.getLeft().getDefiningOp<mlir::db::ConstantOp>();
                  if (!windowConstantOp) {
                     windowConstantOp = distanceOp.getRight().getDefiningOp<mlir::db::ConstantOp>();
                  }
                  if (!windowConstantOp) return;
                  mlir::DenseF32ArrayAttr windowConstantVec = windowConstantOp.getValue().cast<mlir::DenseF32ArrayAttr>();
                  mlir::DenseF32ArrayAttr rangeConstantVec = annRangeSelectionOp.getQueryAttr();
                  if (windowConstantVec != rangeConstantVec) return;
                  if (auto k = findKInSelection(windowOp)) {
                     windowOp->setAttr("FilteredK", mlir::UnitAttr::get(windowOp->getContext()));
                     windowOp->setAttr("needPartition", mlir::UnitAttr::get(windowOp->getContext()));
                     annRangeSelectionOp->setAttr("k", k);
                     annRangeSelectionOp.setComputedColAttr(mapOp.getComputedCols()[0].cast<mlir::tuples::ColumnDefAttr>());
                     mapOp->replaceAllUsesWith(mlir::ValueRange{mapOp.getRel()});
                     toErase.push_back(mapOp.getOperation());
                  }
               }
            }
         }
      });

      for (auto* op : toErase) {
         op->erase();
      }
   }
};
} // end anonymous namespace

namespace mlir { namespace relalg {
std::unique_ptr<Pass> createPartitionByWithANNPass() { return std::make_unique<PartitionByWithANN>(); }
} // end namespace relalg
} // end namespace mlir