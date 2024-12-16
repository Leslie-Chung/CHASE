#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class ExtractNestedOperators : public mlir::PassWrapper<ExtractNestedOperators, mlir::OperationPass<mlir::func::FuncOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractNestedOperators)
   virtual llvm::StringRef getArgument() const override { return "relalg-extract-nested-operators"; }

   void sanitizeOp(mlir::IRMapping& mapping, mlir::Operation* op) const {
      for (size_t i = 0; i < op->getNumOperands(); i++) {
         mlir::Value v = op->getOperand(i);
         if (mapping.contains(v)) {
            op->setOperand(i, mapping.lookup(v));
            continue;
         }
      }
   }
   void runOnOperation() override {
      getOperation().walk([&](Operator innerOperator) {
         // Joinop、SelectionOp、Mapop is a TupleLamdaOperator
         if (auto o = mlir::dyn_cast_or_null<TupleLamdaOperator>(innerOperator->getParentOfType<Operator>().getOperation())) {
            mlir::IRMapping mapping;
            TupleLamdaOperator toMoveBefore;
            while (o) {
               if (auto innerLambda = mlir::dyn_cast_or_null<TupleLamdaOperator>(innerOperator.getOperation())) {
                  mapping.map(o.getLambdaArgument(), innerLambda.getLambdaArgument());
               }
               toMoveBefore = o;
               o = mlir::dyn_cast_or_null<TupleLamdaOperator>(o->getParentOfType<Operator>().getOperation());
            } 
            innerOperator->walk([&](mlir::Operation* op) {
               if (!mlir::isa<Operator>(op)&&op->getParentOp()==innerOperator.getOperation()) {
                  mlir::relalg::detail::inlineOpIntoBlock(op, toMoveBefore, op->getBlock(), mapping);
                  sanitizeOp(mapping, op);
               }
            });
            innerOperator->moveBefore(toMoveBefore);
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createExtractNestedOperatorsPass() { return std::make_unique<ExtractNestedOperators>(); }
} // end namespace relalg
} // end namespace mlir