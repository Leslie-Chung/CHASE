#include "llvm/ADT/TypeSwitch.h"
#include "mlir-support/eval.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stack"
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>

namespace {
class HashJoinUtils {
   public:
   static bool isAndedResult(mlir::Operation* op, bool first = true) {
      if (mlir::isa<mlir::tuples::ReturnOp>(op)) {
         return true;
      }
      if (mlir::isa<mlir::db::AndOp>(op) || first) {
         for (auto* user : op->getUsers()) {
            if (!isAndedResult(user, false)) return false;
         }
         return true;
      } else {
         return false;
      }
   }
   struct MapBlockInfo {
      std::vector<mlir::Value> results;
      mlir::Block* block;
      std::vector<mlir::Attribute> createdColumns;
   };
   static std::pair<std::vector<mlir::Attribute>, std::vector<mlir::Attribute>> extractKeys(mlir::Block* block, mlir::relalg::ColumnSet keyAttributes, mlir::relalg::ColumnSet otherAttributes, MapBlockInfo& mapBlockInfo) {
      std::vector<mlir::Attribute> toHash;
      std::vector<mlir::Attribute> nullsEqual;
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      mlir::IRMapping mapping;
      mapping.map(block->getArgument(0), mapBlockInfo.block->getArgument(0)); // left_tuple+right_tuple
      size_t i = 0;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred(true) && isAndedResult(op)) { 
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               mlir::Value keyVal;
               if (leftAttributes.isSubsetOf(keyAttributes) && rightAttributes.isSubsetOf(otherAttributes)) {
                  keyVal = cmpOp.getLeft();
               } else if (rightAttributes.isSubsetOf(keyAttributes) && leftAttributes.isSubsetOf(otherAttributes)) {
                  keyVal = cmpOp.getRight();
               }
               if (keyVal) { 
                  if (auto getColOp = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(keyVal.getDefiningOp())) {
                     toHash.push_back(getColOp.getAttr());
                  } else { 
                     //todo: remove nasty hack:
                     mlir::OpBuilder builder(cmpOp->getContext());
                     builder.setInsertionPointToEnd(mapBlockInfo.block);
                     auto helperOp = builder.create<mlir::arith::ConstantOp>(cmpOp.getLoc(), builder.getIndexAttr(0));

                     mlir::relalg::detail::inlineOpIntoBlock(keyVal.getDefiningOp(), keyVal.getDefiningOp()->getParentOp(), mapBlockInfo.block, mapping, helperOp);
                     helperOp->remove();
                     helperOp->destroy();

                     auto& colManager = builder.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
                     auto def = colManager.createDef(colManager.getUniqueScope("join"), "key" + std::to_string(i++)); 
                     def.getColumn().type = keyVal.getType();
                     auto ref = colManager.createRef(&def.getColumn());
                     mapBlockInfo.createdColumns.push_back(def);
                     mapBlockInfo.results.push_back(mapping.lookupOrNull(keyVal));
                     toHash.push_back(ref);
                     {
                        mlir::OpBuilder builder2(cmpOp->getContext());
                        builder2.setInsertionPointToStart(block);
                        keyVal.replaceAllUsesWith(builder2.create<mlir::tuples::GetColumnOp>(builder2.getUnknownLoc(), keyVal.getType(), ref, block->getArgument(0)));
                     }
                  }
                  mlir::OpBuilder builder2(cmpOp->getContext());
                  nullsEqual.push_back(builder2.getI8IntegerAttr(!cmpOp.isEqualityPred(false))); 
                  builder2.setInsertionPoint(cmpOp);
                  mlir::Value constTrue = builder2.create<mlir::arith::ConstantIntOp>(builder2.getUnknownLoc(), 1, 1);
                  if (cmpOp->getResult(0).getType().isa<mlir::db::NullableType>()) {
                     constTrue = builder2.create<mlir::db::AsNullableOp>(builder2.getUnknownLoc(), cmpOp->getResult(0).getType(), constTrue);
                  }
                  cmpOp->replaceAllUsesWith(mlir::ValueRange{constTrue}); 
               }
            }
         } else {
            mlir::relalg::ColumnSet attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return {toHash, nullsEqual};
   }
};

class OptimizeImplementations : public mlir::PassWrapper<OptimizeImplementations, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-optimize-implementations"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeImplementations)
   bool hashImplPossible(mlir::Block* block, mlir::relalg::ColumnSet availableLeft, mlir::relalg::ColumnSet availableRight) { //todo: does not work always
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required; 
      mlir::relalg::ColumnSet leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      bool res = false;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred(true) && HashJoinUtils::isAndedResult(op)) {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  res = true;
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  res = true;
               }
            }
         } else {
            mlir::relalg::ColumnSet attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) { 
               required.insert({result, attributes});
            }
         }
      });
      return res;
   }

   // Verify that the join predicate in block takes all and only the primary key columns from baseTableOp into consideration
   bool containsExactlyPrimaryKey(mlir::MLIRContext* ctxt, mlir::Operation* baseTableOp, mlir::Block* block, std::string& indexName) {
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> columns;
      auto baseTable = mlir::cast<mlir::relalg::BaseTableOp>(baseTableOp);
      auto& colManager = ctxt->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

      // Initialize map to verify presence of all primary key attributes
      std::unordered_map<std::string, bool> primaryKeyFound;
      for (auto primaryKeyAttribute : baseTable.getMeta().getMeta()->getPrimaryKey()) {
         primaryKeyFound[primaryKeyAttribute] = false;
      }

      // Verify all cmp operations
      bool res = true;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            columns.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
            mlir::relalg::ColumnSet relevantColumns = columns[cmpOp.getLeft()];
            relevantColumns.insert(columns[cmpOp.getRight()]);
            for (auto* relevantColumn : relevantColumns) {
               std::string tableName = colManager.getName(relevantColumn).first;
               std::string columnName = colManager.getName(relevantColumn).second;

               // Only take columns contained in baseTableOp into consideration
               if (baseTable.getCreatedColumns().contains(relevantColumn)) {
                  // Check that no non-primary key attribute was used
                  if (!primaryKeyFound.contains(columnName)) res = false;
                  // Mark primary key attribute as used
                  else
                     primaryKeyFound[columnName] = true;
               }
            }
         }
      });
      // Check if all primary key attributes were found
      for (auto primaryKeyAttribute : primaryKeyFound) {
         res &= primaryKeyAttribute.second;
      }
      indexName = "pk_hash";
      return res;
   }

   bool annImplPossible(mlir::Block* block) {
      bool res = false;
      block->walk([&](mlir::relalg::CmpOpInterface cmpOp) {
         mlir::Value distanceOperand = cmpOp.getLeft();
         // if (cmpOp.isLessPred(true) || cmpOp.isLessPred(false)) {
         //    distanceOperand = cmpOp.getLeft();
         // } else if (cmpOp.isGreaterPred(true) || cmpOp.isGreaterPred(false)) {
         //    distanceOperand = cmpOp.getRight();
         // }
         if (cmpOp.isGreaterPred(true) || cmpOp.isGreaterPred(false)) {
            distanceOperand = cmpOp.getRight();
         }
         if (distanceOperand.getDefiningOp<mlir::relalg::DistanceOpInterface>()) {
            res = true;
         }
      });
      return res;
   }

   bool containsANNIndexForRange(mlir::MLIRContext* ctxt, mlir::Operation* baseTableOp, mlir::Block* block, std::string& indexName) {
      bool res = false;
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> columns;
      auto baseTable = mlir::cast<mlir::relalg::BaseTableOp>(baseTableOp);
      auto& colManager = ctxt->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      // Initialize map to verify presence of all primary key attributes
      std::unordered_map<std::string, bool> ANNKeyFound;
      for (auto indiceMetaData : baseTable.getMeta().getMeta()->getIndices()) {
         std::string col = indiceMetaData->columns[0];
         if (indiceMetaData->type == runtime::Index::Type::HNSW) {
            ANNKeyFound[col] = true;
         }
      }

      // Verify all cmp operations

      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            columns.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto ditanceOp = mlir::dyn_cast_or_null<mlir::relalg::DistanceOpInterface>(op)) {
            mlir::relalg::ColumnSet relevantColumns = columns[ditanceOp.getLeft()];
            relevantColumns.insert(columns[ditanceOp.getRight()]); 

            for (auto* relevantColumn : relevantColumns) {
               std::string tableName = colManager.getName(relevantColumn).first;
               std::string columnName = colManager.getName(relevantColumn).second;

               // Only take columns contained in baseTableOp into consideration
               if (baseTable.getCreatedColumns().contains(relevantColumn)) { 
                  // Check that no non-primary key attribute was used
                  if (!ANNKeyFound.contains(columnName)) res = false;
                  // Mark primary key attribute as used
                  else {
                     res = true;
                     indexName = columnName;
                     break;
                  }
               }
            }
         }
      });

      return res;
   }

   bool isBaseRelationWithSelects(Operator op, std::stack<mlir::Operation*>& path) {
      // Saves operations until base relation is reached on stack for easy access
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op.getOperation())
         .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp baseTableOp) {
            path.push(baseTableOp.getOperation());
            return true;
         })
         .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp selectionOp) {
            path.push(selectionOp.getOperation());
            for (auto& child : selectionOp.getChildren()) {
               if (!isBaseRelationWithSelects(mlir::cast<Operator>(child.getOperation()), path)) return false;
            }
            return true;
         })
         .Default([&](auto&) {
            return false;
         });
   }

   void prepareForHash(PredicateOperator predicateOperator) {
      auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
      auto left = mlir::cast<Operator>(binOp.leftChild());
      auto right = mlir::cast<Operator>(binOp.rightChild());
      //left
      {
         mlir::OpBuilder builder(&this->getContext());
         HashJoinUtils::MapBlockInfo mapBlockInfo;
         mapBlockInfo.block = new mlir::Block;
         mapBlockInfo.block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto [keys, nullsEqual] = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), left.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.getPredicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<mlir::tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
            left = mapOp;
         }
         predicateOperator->setAttr("leftHash", builder.getArrayAttr(keys));
         predicateOperator->setAttr("nullsEqual", builder.getArrayAttr(nullsEqual));
      }

      //right
      {
         mlir::OpBuilder builder(&this->getContext());
         HashJoinUtils::MapBlockInfo mapBlockInfo;
         mapBlockInfo.block = new mlir::Block;
         mapBlockInfo.block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto [keys, nullEquals] = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), right.getAvailableColumns(), left.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), right.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.getPredicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<mlir::tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
            right = mapOp;
         }
         predicateOperator->setAttr("rightHash", builder.getArrayAttr(keys));
      }
      mlir::cast<Operator>(predicateOperator.getOperation()).setChildren({left, right});
   }

   std::vector<mlir::Attribute> getANNColumn(mlir::Block* block, mlir::relalg::ColumnSet availableLeft, mlir::relalg::ColumnSet availableRight) {
      std::vector<mlir::Attribute> toANN;
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
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

            auto distanceOp = distanceOperand.getDefiningOp<mlir::relalg::DistanceOpInterface>();
            auto constantOp = constantOperand.getDefiningOp<mlir::db::ConstantOp>();
            if (distanceOp && HashJoinUtils::isAndedResult(op)) {
               auto leftAttributes = required[distanceOp.getLeft()];
               auto rightAttributes = required[distanceOp.getRight()];
               mlir::Value keyVal; // getColumn  @XX::@vec
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) { 
                  keyVal = distanceOp.getLeft();
               } else if (rightAttributes.isSubsetOf(availableLeft) && leftAttributes.isSubsetOf(availableRight)) { 
                  keyVal = distanceOp.getRight();
               }
               if (keyVal) { 
                  if (auto getColOp = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(keyVal.getDefiningOp())) { 
                     mlir::OpBuilder builder(&this->getContext());
                     toANN.push_back(getColOp.getAttr()); // column
                     toANN.push_back(builder.getI16IntegerAttr(getBaseType(getColOp.getAttr().getColumn().type).dyn_cast<mlir::db::DenseVectorType>().getSize())); // dim
                     toANN.push_back(builder.getI8IntegerAttr(distanceOp.getDistanceSpace())); // distance space
                     if (auto stringAttr = constantOp.getValue().dyn_cast_or_null<mlir::StringAttr>()) {
                        toANN.push_back(builder.getF32FloatAttr(std::stof(stringAttr.str()))); // range constant
                     } else {
                        toANN.push_back(constantOp.getValue()); // range constant
                     }
                     toANN.push_back(builder.getI32IntegerAttr(-1)); // k constant
                     return;
                  }
               }
            }
         } else {
            mlir::relalg::ColumnSet attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return toANN;
   }

   void prepareForRangeANN(PredicateOperator predicateOperator) {
      auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
      auto left = mlir::cast<Operator>(binOp.leftChild());
      auto right = mlir::cast<Operator>(binOp.rightChild());
      {
         mlir::OpBuilder builder(&this->getContext());

         auto leftANN = getANNColumn(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns());
         auto rightANN = getANNColumn(&predicateOperator.getPredicateBlock(), right.getAvailableColumns(), left.getAvailableColumns());
         predicateOperator->setAttr("leftHash", builder.getArrayAttr(leftANN));
         predicateOperator->setAttr("nullsEqual", builder.getArrayAttr({}));
         predicateOperator->setAttr("rightHash", builder.getArrayAttr(rightANN));
      }
   }
   static mlir::Value mapColsToNullable(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, size_t exisingOffset = 0, mlir::relalg::ColumnSet excluded = {}) {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      std::vector<mlir::Attribute> defAttrs;
      auto* mapBlock = new mlir::Block;
      auto tupleArg = mapBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         std::vector<mlir::Value> res;
         for (mlir::Attribute attr : mapping) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
            auto* defAttr = &relationDefAttr.getColumn();
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[exisingOffset].cast<mlir::tuples::ColumnRefAttr>();
            if (excluded.contains(&fromExisting.getColumn())) continue;
            mlir::Value value = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), fromExisting, tupleArg);
            if (fromExisting.getColumn().type != defAttr->type) {
               mlir::Value tmp = rewriter.create<mlir::db::AsNullableOp>(loc, defAttr->type, value);
               value = tmp;
            }
            res.push_back(value);
            defAttrs.push_back(colManager.createDef(defAttr));
         }
         rewriter.create<mlir::tuples::ReturnOp>(loc, res);
      }
      auto mapOp = rewriter.create<mlir::relalg::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs));
      mapOp.getPredicate().push_back(mapBlock);
      return mapOp.getResult();
   }

   size_t estimatedEvaluationCost(mlir::Value v) {
      if (auto* definingOp = v.getDefiningOp()) {
         return llvm::TypeSwitch<mlir::Operation*, size_t>(definingOp)
            .Case([&](mlir::db::ConstantOp) {
               return 0;
            })
            .Case([&](mlir::db::OrOp orOp) {
               size_t res = orOp.getVals().size();
               for (auto val : orOp.getVals()) {
                  res += estimatedEvaluationCost(val);
               }
               return res;
            })
            .Case([&](mlir::tuples::GetColumnOp& getColumnOp) {
               if (getBaseType(getColumnOp.getType()).isa<mlir::db::StringType>()) {
                  return 4;
               } else if (getBaseType(getColumnOp.getType()).isa<mlir::db::DecimalType>()) {
                  return 2;
               } else {
                  return 1;
               }
            })
            .Case([&](mlir::relalg::CmpOpInterface cmpOp) {
               auto t = cmpOp.getLeft().getType();
               auto childCost = estimatedEvaluationCost(cmpOp.getLeft()) + estimatedEvaluationCost(cmpOp.getRight());
               if (getBaseType(t).isa<mlir::db::StringType>()) {
                  return 10 + childCost;
               } else {
                  return 1 + childCost;
               }
            })
            .Case([&](mlir::db::BetweenOp cmpOp) {
               auto t = cmpOp.getLower().getType();
               auto childCost = estimatedEvaluationCost(cmpOp.getVal()) + estimatedEvaluationCost(cmpOp.getLower()) + estimatedEvaluationCost(cmpOp.getUpper());
               if (getBaseType(t).isa<mlir::db::StringType>()) {
                  return 20 + childCost;
               } else {
                  return 2 + childCost;
               }
            })
            .Case([&](mlir::relalg::DistanceOpInterface distanceOp) { // support vector
               uint32_t size = getBaseType(distanceOp.getLeft().getType()).dyn_cast<mlir::db::DenseVectorType>().getSize();
               return size * 2;
            })
            .Default([&](mlir::Operation* op) {
               return 1000;
            });
      } else {
         return 10000;
      }
   }
   static mlir::relalg::ColumnSet getRequired(Operator op) {
      auto available = op.getAvailableColumns();

      mlir::relalg::ColumnSet required;
      for (auto* user : op->getUsers()) {
         if (auto consumingOp = mlir::dyn_cast_or_null<Operator>(user)) {
            required.insert(getRequired(consumingOp));
            required.insert(consumingOp.getUsedColumns());
         }
         if (auto materializeOp = mlir::dyn_cast_or_null<mlir::relalg::MaterializeOp>(user)) {
            required.insert(mlir::relalg::ColumnSet::fromArrayAttr(materializeOp.getCols()));
         }
      }
      return available.intersect(required);
   }

   void runOnOperation() override {
      std::vector<mlir::Operation*> toErase;

      getOperation().walk([&](Operator op) {
         ::llvm::TypeSwitch<mlir::Operation*, void>(op.getOperation())
            .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp selectionOp) {
               std::unordered_set<mlir::Operation*> users(selectionOp->getUsers().begin(), selectionOp->getUsers().end());
               if (users.empty()) return;
               bool reorder = users.size() > 1 || !mlir::isa<mlir::relalg::SelectionOp>(*users.begin());
               if (!reorder) return;
               std::vector<mlir::relalg::SelectionOp> selections;
               selections.push_back(selectionOp);
               mlir::relalg::SelectionOp currentSelection = selectionOp;
               while (currentSelection) { 
                  Operator child = currentSelection.getChildren()[0];
                  if (std::vector<mlir::Operation*>(child->getUsers().begin(), child->getUsers().end()).size() > 1) {
                     child = {};
                  }
                  currentSelection = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(child.getOperation());
                  if (currentSelection) {
                     selections.push_back(currentSelection);
                  }
               }
               auto firstStream = selections[selections.size() - 1].getRel(); 
               mlir::relalg::BaseTableOp baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(selections[selections.size() - 1].getRel().getDefiningOp());
               if (baseTableOp) { 
                  std::unordered_map<const mlir::tuples::Column*, std::string> mapping;
                  for (auto c : baseTableOp.getColumns()) {
                     mapping[&c.getValue().cast<mlir::tuples::ColumnDefAttr>().getColumn()] = c.getName().str();
                  }
                  auto meta = baseTableOp.getMeta().getMeta();
                  auto sample = meta->getSample();
                  if (sample) {
                     for (auto selOp : selections) {
                        auto v = mlir::cast<mlir::tuples::ReturnOp>(selOp.getPredicateBlock().getTerminator()).getResults()[0];
                        auto expr = mlir::relalg::buildEvalExpr(v, mapping);
                        auto optionalCount = support::eval::countResults(sample, std::move(expr));
                        if (optionalCount) {
                           auto count = optionalCount.value();
                           if (count == 0) count = 1;
                           double selectivity = static_cast<double>(count) / static_cast<double>(sample->num_rows());
                           selOp->setAttr("selectivity", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), selectivity));
                        }
                     }
                  }
               }
               for (auto selOp : selections) {
                  auto v = mlir::cast<mlir::tuples::ReturnOp>(selOp.getPredicateBlock().getTerminator()).getResults()[0];
                  double evaluationCost = estimatedEvaluationCost(v);
                  selOp->setAttr("evaluationCost", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), evaluationCost));
               }
               if (selections.size() > 1) {
                  std::vector<mlir::relalg::SelectionOp> finalOrder;
                  std::vector<std::pair<double, mlir::relalg::SelectionOp>> toProcess;
                  for (auto sel : selections) {
                     double selectivity = sel->hasAttr("selectivity") ? sel->getAttrOfType<mlir::FloatAttr>("selectivity").getValueAsDouble() : 1;
                     toProcess.push_back({selectivity, sel});
                  }
                  std::sort(toProcess.begin(), toProcess.end(), [](auto a, auto b) { return a.first < b.first; });
                  for (size_t i = 0; i < toProcess.size() - 1; i++) {
                     auto& first = toProcess[i];
                     auto& second = toProcess[i + 1];
                     auto s1 = first.first;
                     auto s2 = second.first;
                     auto c1 = first.second->getAttrOfType<mlir::FloatAttr>("evaluationCost").getValueAsDouble();
                     auto c2 = second.second->getAttrOfType<mlir::FloatAttr>("evaluationCost").getValueAsDouble();
                     if (c1 + s1 * c2 > c2 + s2 * c1) { // cost1 + sel1 * cost2  >  cost2 + sel2 * cost1
                        std::swap(first, second); 
                     }
                  }
                  mlir::Value stream = firstStream;
                  for (auto sel : toProcess) {
                     sel.second->moveAfter(stream.getDefiningOp());
                     sel.second.setOperand(stream);
                     stream = sel.second;
                  }
                  selectionOp.getResult().replaceUsesWithIf(stream, [&](auto& use) {
                     return users.contains(use.getOwner());
                  });
               }
            })
            .Case<mlir::relalg::LimitOp>([&](mlir::relalg::LimitOp limitOp) {
               if (auto sortOp = mlir::dyn_cast_or_null<mlir::relalg::SortOp>(limitOp.getRel().getDefiningOp())) {
                  mlir::OpBuilder builder(limitOp);
                  toErase.push_back(limitOp);
                  toErase.push_back(sortOp);

                  limitOp.replaceAllUsesWith(builder.create<mlir::relalg::TopKOp>(limitOp.getLoc(), limitOp.getMaxRows(), sortOp.getRel(), sortOp.getSortspecs()).asRelation());
               }
            })
            .Case<mlir::relalg::InnerJoinOp, mlir::relalg::CollectionJoinOp, mlir::relalg::FullOuterJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (annImplPossible(&predicateOperator.getPredicateBlock())) {
                  std::stack<mlir::Operation*> leftPath, rightPath;
                  std::string leftIndexName, rightIndexName;
                  bool leftCanUseANNIndex = isBaseRelationWithSelects(left, leftPath) && containsANNIndexForRange(binOp.getContext(), leftPath.top(), &predicateOperator.getPredicateBlock(), leftIndexName);
                  bool rightCanUseANNIndex = isBaseRelationWithSelects(right, rightPath) && containsANNIndexForRange(binOp.getContext(), rightPath.top(), &predicateOperator.getPredicateBlock(), rightIndexName);
                  // Select possible build side to the left
                  bool isInnerJoin = mlir::isa<mlir::relalg::InnerJoinOp>(predicateOperator);
                  bool reversed = false;

                  if (isInnerJoin && (leftCanUseANNIndex || rightCanUseANNIndex)) {
                     prepareForRangeANN(predicateOperator);

                     if (leftCanUseANNIndex && rightCanUseANNIndex) {
                        // Compute heuristic of which base table the index is more beneficial
                        // Used heuristic: prefer bigger ratio of |buildSide| / |probeSide|
                        auto leftBaseTable = mlir::cast<mlir::relalg::BaseTableOp>(leftPath.top());
                        auto rightBaseTable = mlir::cast<mlir::relalg::BaseTableOp>(rightPath.top());
                        int numBaseRowsLeft = leftBaseTable.getMeta().getMeta()->getNumRows() + 1;
                        int numBaseRowsRight = rightBaseTable.getMeta().getMeta()->getNumRows() + 1;
                        int numNonBaseRowsLeft = left->hasAttr("rows") ? left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>().getValueAsDouble() + 1 : 1;
                        int numNonBaseRowsRight = right->hasAttr("rows") ? right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>().getValueAsDouble() + 1 : 1;
                        // Exchange left and right side if deemed beneficial by heuristic
                        if (numNonBaseRowsRight / numBaseRowsLeft < numNonBaseRowsLeft / numBaseRowsRight) {
                           reversed = true;
                           std::swap(left, right);
                           std::swap(leftPath, rightPath);
                           std::swap(leftIndexName, rightIndexName);
                           mlir::Attribute tmp = predicateOperator->getAttr("rightHash");
                           predicateOperator->setAttr("rightHash", predicateOperator->getAttr("leftHash"));
                           predicateOperator->setAttr("leftHash", tmp);
                        }
                     } else if (!leftCanUseANNIndex) {
                        // Exchange left and right side
                        reversed = true;
                        std::swap(left, right);
                        std::swap(leftPath, rightPath);
                        std::swap(leftIndexName, rightIndexName);
                        leftCanUseANNIndex = true;
                        mlir::Attribute tmp = predicateOperator->getAttr("rightHash");
                        predicateOperator->setAttr("rightHash", predicateOperator->getAttr("leftHash"));
                        predicateOperator->setAttr("leftHash", tmp);
                     }
                     if (isInnerJoin && leftCanUseANNIndex) {
                        predicateOperator.getPredicateBlock().walk([&](mlir::relalg::CmpOpInterface cmpOp) {
                           mlir::Value distanceOperand = cmpOp.getLeft();
                           // if (cmpOp.isLessPred(true) || cmpOp.isLessPred(false)) {
                           //    distanceOperand = cmpOp.getLeft();
                           // } else if (cmpOp.isGreaterPred(true) || cmpOp.isGreaterPred(false)) {
                           //    distanceOperand = cmpOp.getRight();
                           // }
                           if (cmpOp.isGreaterPred(true) || cmpOp.isGreaterPred(false)) {
                              distanceOperand = cmpOp.getRight();
                           }
                           auto distanceOp = distanceOperand.getDefiningOp<mlir::relalg::DistanceOpInterface>();
                           if (distanceOp && HashJoinUtils::isAndedResult(cmpOp)) {
                              mlir::OpBuilder builder(cmpOp);
                              // toErase.push_back(distanceOp.getLeft().getDefiningOp()); // getColumn
                              // toErase.push_back(distanceOp.getRight().getDefiningOp());
                              // toErase.push_back(cmpOp.getLeft().getDefiningOp());
                              // toErase.push_back(cmpOp.getRight().getDefiningOp());
                              // toErase.push_back(cmpOp);
                              mlir::Value constTrue = builder.create<mlir::arith::ConstantIntOp>(cmpOp->getLoc(), 1, 1);
                              if (cmpOp->getResult(0).getType().isa<mlir::db::NullableType>()) {
                                 constTrue = builder.create<mlir::db::AsNullableOp>(cmpOp->getLoc(), cmpOp->getResult(0).getType(), constTrue);
                              }
                              cmpOp->replaceAllUsesWith(mlir::ValueRange{constTrue}); 
                           }
                        });

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
                        leftBaseTable->setAttr("virtual", mlir::UnitAttr::get(&getContext()));

                        // Add name of table to leftHash annotation
                        std::vector<mlir::Attribute> leftANN;
                        leftANN.push_back(leftBaseTable.getTableIdentifierAttr());
                        leftANN.push_back(mlir::StringAttr::get(op.getContext(), leftIndexName));
                        for (auto attr : op->getAttr("leftHash").dyn_cast_or_null<mlir::ArrayAttr>()) {
                           leftANN.push_back(attr);
                        }
                        op->setAttr("leftHash", mlir::ArrayAttr::get(&getContext(), leftANN));

                        // std::vector<mlir::Attribute> rightANN;
                        // rightANN.push_back(leftBaseTable.getTableIdentifierAttr());
                        // rightANN.push_back(mlir::StringAttr::get(op.getContext(), leftIndexName));
                        // for (auto attr : op->getAttr("leftHash").dyn_cast_or_null<mlir::ArrayAttr>()) {
                        //    leftANN.push_back(attr);
                        // }

                        // op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "annIndexNestedLoop"));
                        op->setAttr("useANNIndexNestedLoop", mlir::UnitAttr::get(op.getContext()));
                        // op->setAttr("index", mlir::StringAttr::get(op.getContext(), leftIndexName));
                     }
                     return;
                  }
               }
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  // Determine if index nested loop is possible and is beneficial
                  std::stack<mlir::Operation*> leftPath, rightPath;
                  std::string leftIndexName, rightIndexName;
                  bool leftCanUsePrimaryKeyIndex = isBaseRelationWithSelects(left, leftPath) && containsExactlyPrimaryKey(binOp.getContext(), leftPath.top(), &predicateOperator.getPredicateBlock(), leftIndexName);
                  bool rightCanUsePrimaryKeyIndex = isBaseRelationWithSelects(right, rightPath) && containsExactlyPrimaryKey(binOp.getContext(), rightPath.top(), &predicateOperator.getPredicateBlock(), rightIndexName);
                  bool isInnerJoin = mlir::isa<mlir::relalg::InnerJoinOp>(predicateOperator);
                  bool reversed = false;

                  prepareForHash(predicateOperator);

                  // Select possible build side to the left
                  if (isInnerJoin && (leftCanUsePrimaryKeyIndex || rightCanUsePrimaryKeyIndex)) {
                     if (leftCanUsePrimaryKeyIndex && rightCanUsePrimaryKeyIndex) {
                        // Compute heuristic of which base table the index is more beneficial
                        // Used heuristic: prefer bigger ratio of |buildSide| / |probeSide|
                        auto leftBaseTable = mlir::cast<mlir::relalg::BaseTableOp>(leftPath.top());
                        auto rightBaseTable = mlir::cast<mlir::relalg::BaseTableOp>(rightPath.top());
                        int numBaseRowsLeft = leftBaseTable.getMeta().getMeta()->getNumRows() + 1;
                        int numBaseRowsRight = rightBaseTable.getMeta().getMeta()->getNumRows() + 1;
                        int numNonBaseRowsLeft = left->hasAttr("rows") ? left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>().getValueAsDouble() + 1 : 1;
                        int numNonBaseRowsRight = right->hasAttr("rows") ? right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>().getValueAsDouble() + 1 : 1;
                        // Exchange left and right side if deemed beneficial by heuristic
                        if (numNonBaseRowsRight / numBaseRowsLeft < numNonBaseRowsLeft / numBaseRowsRight) {
                           reversed = true;
                           std::swap(left, right);
                           std::swap(leftPath, rightPath);
                           std::swap(leftIndexName, rightIndexName);
                           mlir::Attribute tmp = predicateOperator->getAttr("rightHash");
                           predicateOperator->setAttr("rightHash", predicateOperator->getAttr("leftHash"));
                           predicateOperator->setAttr("leftHash", tmp);
                        }
                     } else if (!leftCanUsePrimaryKeyIndex) {
                        // Exchange left and right side
                        reversed = true;
                        std::swap(left, right);
                        std::swap(leftPath, rightPath);
                        std::swap(leftIndexName, rightIndexName);
                        leftCanUsePrimaryKeyIndex = true;
                        mlir::Attribute tmp = predicateOperator->getAttr("rightHash");
                        predicateOperator->setAttr("rightHash", predicateOperator->getAttr("leftHash"));
                        predicateOperator->setAttr("leftHash", tmp);
                     }
                  }

                  // Compute correct number of rows for index nested loop join
                  double numRowsLeft = 0, numRowsRight = std::numeric_limits<double>::max(); // default: disable inlj
                  if (auto leftCardinalityAttr = left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                     numRowsLeft = leftCardinalityAttr.getValueAsDouble();
                  }
                  if (auto rightCardinalityAttr = right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                     numRowsRight = rightCardinalityAttr.getValueAsDouble();
                  }
                  if (isInnerJoin && leftCanUsePrimaryKeyIndex && right->hasAttr("rows") && 20 * numRowsRight < numRowsLeft) {
                     // base relations do not need to be moved
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
                     leftBaseTable->setAttr("virtual", mlir::UnitAttr::get(&getContext()));

                     // Add name of table to leftHash annotation
                     std::vector<mlir::Attribute> leftHash;
                     leftHash.push_back(leftBaseTable.getTableIdentifierAttr());
                     for (auto attr : op->getAttr("leftHash").dyn_cast_or_null<mlir::ArrayAttr>()) {
                        leftHash.push_back(attr);
                     }
                     op->setAttr("leftHash", mlir::ArrayAttr::get(&getContext(), leftHash));

                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "indexNestedLoop"));
                     op->setAttr("useIndexNestedLoop", mlir::UnitAttr::get(op.getContext()));
                     op->setAttr("index", mlir::StringAttr::get(op.getContext(), leftIndexName));
                  } else {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                     op->setAttr("useHashJoin", mlir::UnitAttr::get(op.getContext()));
                     prepareForHash(predicateOperator);
                  }
               }
            })
            .Case<mlir::relalg::SemiJoinOp, mlir::relalg::AntiSemiJoinOp, mlir::relalg::OuterJoinOp, mlir::relalg::MarkJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (left->hasAttr("rows") && right->hasAttr("rows")) {
                  double rowsLeft = 0;
                  double rowsRight = 0;
                  if (auto lDAttr = left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                     rowsLeft = lDAttr.getValueAsDouble();
                  } else if (auto lIAttr = left->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                     rowsLeft = lIAttr.getInt();
                  }
                  if (auto rDAttr = right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                     rowsRight = rDAttr.getValueAsDouble();
                  } else if (auto rIAttr = right->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                     rowsRight = rIAttr.getInt();
                  }
                  if (rowsLeft < rowsRight) {
                     op->setAttr("reverseSides", mlir::UnitAttr::get(op.getContext()));
                  }
               }
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  op->setAttr("useHashJoin", mlir::UnitAttr::get(op.getContext()));
                  prepareForHash(predicateOperator);
                  if (left->hasAttr("rows") && right->hasAttr("rows")) {
                     double rowsLeft = 0;
                     double rowsRight = 0;
                     if (auto lDAttr = left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                        rowsLeft = lDAttr.getValueAsDouble();
                     } else if (auto lIAttr = left->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                        rowsLeft = lIAttr.getInt();
                     }
                     if (auto rDAttr = right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                        rowsRight = rDAttr.getValueAsDouble();
                     } else if (auto rIAttr = right->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                        rowsRight = rIAttr.getInt();
                     }
                     if (rowsLeft < rowsRight) {
                        op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "markhash"));
                     } else {
                        op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                     }
                  } else {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                  }
               }
            })
            .Case<mlir::relalg::SingleJoinOp>([&](mlir::relalg::SingleJoinOp op) {
               if (auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicateBlock().getTerminator())) {
                  if (returnOp.getResults().empty()) {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "constant"));
                     op->setAttr("constantJoin", mlir::UnitAttr::get(op.getContext()));
                  }
               }
               auto left = mlir::cast<Operator>(op.leftChild());
               auto right = mlir::cast<Operator>(op.rightChild());
               if (hashImplPossible(&op.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  prepareForHash(op);
                  op->setAttr("useHashJoin", mlir::UnitAttr::get(op.getContext()));
                  op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
               }
            })
            .Case<mlir::relalg::WindowOp>([&](mlir::relalg::WindowOp windowOp) {

            })

            .Default([&](auto x) {
            });
      });
      for (auto* op : toErase) {
         op->erase();
      }
      toErase.clear();
      getOperation().walk([&](mlir::relalg::InnerJoinOp op) {
         auto usedColumns = op.getUsedColumns();
         if (!op->hasAttr("leftHash") || !op->hasAttr("rightHash")) return; 
         auto leftKeys = op->getAttr("leftHash").cast<mlir::ArrayAttr>();
         auto rightKeys = op->getAttr("rightHash").cast<mlir::ArrayAttr>();
         auto leftKeySet = mlir::relalg::ColumnSet::fromArrayAttr(leftKeys);
         auto rightKeySet = mlir::relalg::ColumnSet::fromArrayAttr(rightKeys);
         mlir::relalg::ColumnSet reallyRequiredColumns;
         reallyRequiredColumns.insert(rightKeySet);
         auto currentChild = op.getRight();
         std::vector<mlir::Operation*> otherOps;
         bool needsSplit = false;

         while (true) {
            if (auto aggrOp = mlir::dyn_cast_or_null<mlir::relalg::AggregationOp>(currentChild.getDefiningOp())) {
               if (mlir::relalg::ColumnSet::fromArrayAttr(aggrOp.getComputedCols()).intersects(reallyRequiredColumns)) {
                  return;
               } else {
                  needsSplit |= mlir::relalg::ColumnSet::fromArrayAttr(aggrOp.getComputedCols()).intersects(usedColumns);
                  auto groupByKeySet = mlir::relalg::ColumnSet::fromArrayAttr(aggrOp.getGroupByCols());
                  if (groupByKeySet.size() != leftKeySet.size()) return;
                  groupByKeySet.remove(leftKeySet);
                  groupByKeySet.remove(rightKeySet);
                  if (!groupByKeySet.empty()) return;

                  auto fds = mlir::cast<Operator>(op.getLeft().getDefiningOp()).getFDs();
                  if (!fds.isDuplicateFreeKey(leftKeySet)) return;
                  std::vector<mlir::Operation*> users(op->getUsers().begin(), op->getUsers().end());
                  if (users.size() != 1) {
                     return;
                  }
                  auto required = getRequired(op);
                  mlir::Operation* moveBefore = users[0];
                  for (auto* o : otherOps) {
                     o->moveBefore(moveBefore);
                     moveBefore = o;
                  }
                  auto topRightVal = op.getRight();
                  op.getResult().replaceAllUsesWith(topRightVal);
                  op.setOperand(1, aggrOp.getRel());
                  aggrOp->moveAfter(op);
                  aggrOp->setOperand(0, op.getResult());
                  if (needsSplit) {
                     mlir::OpBuilder builder(&getContext());
                     builder.setInsertionPointAfter(topRightVal.getDefiningOp());
                     auto sel = builder.create<mlir::relalg::SelectionOp>(op->getLoc(), topRightVal);
                     topRightVal.replaceAllUsesExcept(sel.asRelation(), sel);
                     sel.getPredicate().takeBody(op.getPredicate());
                     mlir::Block* newJoinPred = new mlir::Block;
                     newJoinPred->addArgument(mlir::tuples::TupleType::get(builder.getContext()), op.getLoc());
                     builder.setInsertionPointToStart(newJoinPred);
                     mlir::Value trueVal = builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 1);
                     builder.create<mlir::tuples::ReturnOp>(op.getLoc(), trueVal);
                     op.getPredicate().push_back(newJoinPred);
                     required.insert(sel.getUsedColumns());
                  }
                  auto available = mlir::cast<Operator>(topRightVal.getDefiningOp()).getAvailableColumns();
                  required.remove(available);
                  //llvm::dbgs() << "still required:";
                  //required.dump(&getContext());
                  if (!required.empty()) {
                     mlir::OpBuilder builder(&getContext());
                     auto previousTerminator = mlir::cast<mlir::tuples::ReturnOp>(aggrOp.getAggrFunc().front().getTerminator());
                     std::vector<mlir::Value> aggregates(previousTerminator.getResults().begin(), previousTerminator.getResults().end());
                     builder.setInsertionPointToEnd(&aggrOp.getAggrFunc().front());
                     std::vector<mlir::Attribute> computedCols(aggrOp.getComputedCols().begin(), aggrOp.getComputedCols().end());
                     auto& colManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
                     std::vector<mlir::Attribute> renaming;
                     for (auto* column : required) {
                        auto* newCol = colManager.get(colManager.getUniqueScope("moved_aggr"), colManager.getName(column).second).get();
                        newCol->type = column->type;
                        renaming.push_back(colManager.createDef(column, builder.getArrayAttr({colManager.createRef(newCol)})));
                        computedCols.push_back(colManager.createDef(newCol));
                        aggregates.push_back(builder.create<mlir::relalg::AggrFuncOp>(aggrOp->getLoc(), column->type, mlir::relalg::AggrFunc::any, aggrOp.getAggrFunc().getArgument(0), colManager.createRef(column)));
                     }
                     builder.create<mlir::tuples::ReturnOp>(aggrOp.getLoc(), aggregates);
                     previousTerminator->erase();
                     builder.setInsertionPointAfter(aggrOp);
                     auto renamed = builder.create<mlir::relalg::RenamingOp>(aggrOp.getLoc(), aggrOp.getResult(), builder.getArrayAttr(renaming));
                     aggrOp.getResult().replaceAllUsesExcept(renamed.getResult(), renamed);
                     aggrOp.setComputedColsAttr(builder.getArrayAttr(computedCols));
                  }
                  return;
               }
            } else if (auto selOp = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(currentChild.getDefiningOp())) {
               otherOps.push_back(selOp.getOperation());
               currentChild = selOp.getRel();
            } else if (auto mapOp = mlir::dyn_cast_or_null<mlir::relalg::MapOp>(currentChild.getDefiningOp())) {
               if (mlir::relalg::ColumnSet::fromArrayAttr(mapOp.getComputedCols()).intersects(reallyRequiredColumns)) {
                  return;
               } else {
                  needsSplit |= mlir::relalg::ColumnSet::fromArrayAttr(mapOp.getComputedCols()).intersects(usedColumns);
                  otherOps.push_back(mapOp.getOperation());
                  currentChild = mapOp.getRel();
               }
            } else {
               return;
            }
         }
      });
      bool enableGroupJoins = true;
      if (const char* shouldEnableGJ = std::getenv("LINGODB_ENABLE_GJ")) {
         if (std::string(shouldEnableGJ) == "OFF") {
            enableGroupJoins = false;
         }
      }
      if (enableGroupJoins) {
         getOperation().walk([&](mlir::relalg::AggregationOp op) {
            auto* potentialJoin = op.getRel().getDefiningOp();
            mlir::relalg::MapOp mapOp = mlir::dyn_cast_or_null<mlir::relalg::MapOp>(potentialJoin);
            mlir::relalg::ColumnSet usedColumns = op.getUsedColumns();
            if (mapOp) {
               usedColumns.insert(mapOp.getUsedColumns());
               potentialJoin = mapOp.getRel().getDefiningOp();
            }

            auto isInnerJoin = mlir::isa<mlir::relalg::InnerJoinOp>(potentialJoin);
            auto isOuterJoin = mlir::isa<mlir::relalg::OuterJoinOp>(potentialJoin);
            if (!isInnerJoin && !isOuterJoin) return;
            PredicateOperator join = mlir::cast<PredicateOperator>(potentialJoin);
            Operator joinOperator = mlir::cast<Operator>(potentialJoin);
            usedColumns.insert(joinOperator.getUsedColumns());
            if (!join->hasAttr("useHashJoin") || !join->hasAttr("leftHash") || !join->hasAttr("rightHash")) return;
            auto leftKeys = join->getAttr("leftHash").cast<mlir::ArrayAttr>();
            auto rightKeys = join->getAttr("rightHash").cast<mlir::ArrayAttr>();
            for (auto p : llvm::zip(leftKeys, rightKeys)) {
               auto t1 = std::get<0>(p).cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
               auto t2 = std::get<1>(p).cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
               if (t1 != t2) return;
            }
            auto leftKeySet = mlir::relalg::ColumnSet::fromArrayAttr(leftKeys);
            auto rightKeySet = mlir::relalg::ColumnSet::fromArrayAttr(rightKeys);
            auto groupByKeySet = mlir::relalg::ColumnSet::fromArrayAttr(op.getGroupByCols());
            if (groupByKeySet.size() != leftKeySet.size()) return;
            groupByKeySet.remove(leftKeySet);
            groupByKeySet.remove(rightKeySet);
            if (!groupByKeySet.empty()) return;

            auto leftChild = joinOperator.getChildren()[0];
            auto rightChild = joinOperator.getChildren()[1];
            //auto leftUsedColumns = usedColumns.intersect(leftChild.getAvailableColumns());
            auto fds = leftChild.getFDs();
            if (!fds.isDuplicateFreeKey(leftKeySet)) return;
            bool containsProjection = false;
            bool containsCountRows = false;
            op.walk([&](mlir::relalg::ProjectionOp) { containsProjection = true; });
            op.walk([&](mlir::relalg::CountRowsOp) { containsCountRows = true; });
            if (containsProjection || (isOuterJoin && containsCountRows)) return;
            mlir::OpBuilder builder(op);
            mlir::ArrayAttr mappedCols = mapOp ? mapOp.getComputedCols() : builder.getArrayAttr({});
            mlir::Value left = leftChild.asRelation();
            mlir::Value right = rightChild.asRelation();
            if (isOuterJoin) {
               auto outerJoin = mlir::cast<mlir::relalg::OuterJoinOp>(potentialJoin);
               right = mapColsToNullable(right, builder, op.getLoc(), outerJoin.getMapping());
            }
            llvm::dbgs() << "introducing groupjoin\n";
            auto groupJoinOp = builder.create<mlir::relalg::GroupJoinOp>(op.getLoc(), left, right, isOuterJoin ? mlir::relalg::GroupJoinBehavior::outer : mlir::relalg::GroupJoinBehavior::inner, leftKeys, rightKeys, mappedCols, op.getComputedCols());
            if (mapOp) {
               mlir::IRMapping mapping;
               mapOp.getPredicate().cloneInto(&groupJoinOp.getMapFunc(), mapping);
            } else {
               auto* b = new mlir::Block;
               mlir::OpBuilder mB(&getContext());
               groupJoinOp.getMapFunc().push_back(b);
               mB.setInsertionPointToStart(b);
               mB.create<mlir::tuples::ReturnOp>(op.getLoc());
            }
            {
               mlir::IRMapping mapping;
               op.getAggrFunc().cloneInto(&groupJoinOp.getAggrFunc(), mapping);
            }
            {
               mlir::IRMapping mapping;
               join.getPredicateRegion().cloneInto(&groupJoinOp.getPredicate(), mapping);
            }
            op->replaceAllUsesWith(groupJoinOp);
            toErase.push_back(op.getOperation());
            if (mapOp) {
               toErase.push_back(mapOp);
            }
            toErase.push_back(join.getOperation());
         });
      }
      for (auto* op : toErase) {
         op->erase();
      }
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createOptimizeImplementationsPass() { return std::make_unique<OptimizeImplementations>(); }
} // end namespace relalg
} // end namespace mlir