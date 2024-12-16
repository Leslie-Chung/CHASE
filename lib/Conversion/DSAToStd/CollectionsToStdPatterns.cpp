#include "mlir/Conversion/DSAToStd/CollectionIteration.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

class ForOpLowering : public OpConversionPattern<mlir::dsa::ForOp> {
   public:
   using OpConversionPattern<mlir::dsa::ForOp>::OpConversionPattern;
   std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder) const {
      for (size_t i = 0; i < values.size(); i++) {
         values[i] = builder.getRemappedValue(values[i]);
      }
      return values;
   }

   LogicalResult matchAndRewrite(mlir::dsa::ForOp forOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Type> argumentTypes;
      std::vector<Location> argumentLocs;
      for (auto t : forOp.getRegion().getArgumentTypes()) {
         argumentTypes.push_back(t);
         argumentLocs.push_back(forOp->getLoc());
      }
      auto collectionType = forOp.getCollection().getType().dyn_cast_or_null<mlir::util::CollectionType>();
      auto iterator = mlir::dsa::CollectionIterationImpl::getImpl(collectionType, adaptor.getCollection());

      ModuleOp parentModule = forOp->getParentOfType<ModuleOp>();
      std::vector<Value> results = iterator->implementLoop(forOp->getLoc(), adaptor.getInitArgs(), *typeConverter, rewriter, parentModule, [&](std::function<Value(OpBuilder & b)> getElem, ValueRange iterargs, OpBuilder builder) {
         auto yieldOp = cast<mlir::dsa::YieldOp>(forOp.getBody()->getTerminator());
         std::vector<Type> resTypes;
         std::vector<Location> locs;
         for (auto t : yieldOp.getResults()) {
            resTypes.push_back(typeConverter->convertType(t.getType()));
            locs.push_back(forOp->getLoc());
         }
         std::vector<Value> values;
         values.push_back(getElem(builder));
         values.insert(values.end(), iterargs.begin(), iterargs.end());
         auto term = builder.create<mlir::scf::YieldOp>(forOp->getLoc());
         builder.setInsertionPoint(term);
         rewriter.inlineBlockBefore(forOp.getBody(), &*builder.getInsertionPoint(), values);

         std::vector<Value> results(yieldOp.getResults().begin(), yieldOp.getResults().end());
         rewriter.eraseOp(yieldOp);
         rewriter.eraseOp(term);

         return results;
      });
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);

         forOp.getRegion().push_back(new Block());
         forOp.getRegion().front().addArguments(argumentTypes, argumentLocs);
         rewriter.setInsertionPointToStart(&forOp.getRegion().front());
         rewriter.create<mlir::dsa::YieldOp>(forOp.getLoc());
      }

      rewriter.replaceOp(forOp, results);
      return success();
   }
};

class AtLowering : public OpConversionPattern<mlir::dsa::At> {
   public:
   using OpConversionPattern<mlir::dsa::At>::OpConversionPattern;
   static Value getBit(OpBuilder builder, Location loc, Value bits, Value pos) { 
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto indexType = IndexType::get(builder.getContext());
      Value const3 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 3));
      Value const7 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 7));
      Value const1Byte = builder.create<arith::ConstantOp>(loc, i8Type, builder.getIntegerAttr(i8Type, 1));

      Value div8 = builder.create<arith::ShRUIOp>(loc, indexType, pos, const3);
      Value rem8 = builder.create<arith::AndIOp>(loc, indexType, pos, const7);
      Value loadedByte = builder.create<mlir::util::LoadOp>(loc, i8Type, bits, div8);
      Value rem8AsByte = builder.create<arith::IndexCastOp>(loc, i8Type, rem8);
      Value shifted = builder.create<arith::ShRUIOp>(loc, i8Type, loadedByte, rem8AsByte);
      Value res1 = shifted;

      Value anded = builder.create<arith::AndIOp>(loc, i8Type, res1, const1Byte);
      Value res = builder.create<arith::CmpIOp>(loc, i1Type, mlir::arith::CmpIPredicate::eq, anded, const1Byte);
      return res;
   }

   public:
   LogicalResult matchAndRewrite(mlir::dsa::At atOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override { 
      auto loc = atOp->getLoc();
      auto baseType = getBaseType(atOp.getType(0));
      mlir::Value index;
      mlir::Value columnOffset;
      auto indexType = rewriter.getIndexType();
      mlir::Value originalValueBuffer;
      mlir::Value valueBuffer;
      mlir::Value validityBuffer;
      mlir::Value varLenBuffer;
      mlir::Value nullMultiplier;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         if (auto* definingOp = adaptor.getCollection().getDefiningOp()) {
            rewriter.setInsertionPointAfter(definingOp);
         }
         auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.getCollection());
         index = unpacked.getResult(0);
         auto info = unpacked.getResult(1);
         size_t column = atOp.getPos();
         size_t baseOffset = 1 + column * 5; // each column contains the following 5 values
         // columnOffset: Offset where the values for the column begins in the originalValueBuffer
         columnOffset = rewriter.create<mlir::util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset);
         // nullMultiplier: necessary to compute the position of validity bit in validityBuffer
         nullMultiplier = rewriter.create<mlir::util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset + 1);
         // validityBuffer: pointer to bytes to encode invalid values (e.g. null)
         validityBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, info.getType().cast<TupleType>().getType(baseOffset + 2), info, baseOffset + 2);
         // originalValueBuffer: pointer to the location of the row in memory
         originalValueBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, info.getType().cast<TupleType>().getType(baseOffset + 3), info, baseOffset + 3);
         valueBuffer = rewriter.create<mlir::util::ArrayElementPtrOp>(loc, originalValueBuffer.getType(), originalValueBuffer, columnOffset); // pointer to the column
         // varLenBuffer: pointer to variable sized data store
         varLenBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, info.getType().cast<TupleType>().getType(baseOffset + 4), info, baseOffset + 4);
      }
      Value val;
      auto* context = rewriter.getContext();
      if (baseType.isa<util::VarLen32Type>()) {
         Value pos1 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, index);
         pos1.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value const1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         Value ip1 = rewriter.create<arith::AddIOp>(loc, indexType, index, const1);
         Value pos2 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, ip1);
         pos2.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value len = rewriter.create<arith::SubIOp>(loc, rewriter.getI32Type(), pos2, pos1);
         Value pos1AsIndex = rewriter.create<arith::IndexCastOp>(loc, indexType, pos1);
         Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), varLenBuffer, pos1AsIndex);
         val = rewriter.create<mlir::util::CreateVarLen>(loc, mlir::util::VarLen32Type::get(rewriter.getContext()), ptr, len);
      } else if (isIntegerType(baseType, 1)) {
         Value realPos = rewriter.create<arith::AddIOp>(loc, indexType, columnOffset, index);
         val = getBit(rewriter, loc, originalValueBuffer, realPos);
      } else if (typeConverter->convertType(baseType).isIntOrIndexOrFloat()) {
         auto convertedType = typeConverter->convertType(baseType);
         if (auto numBytesAttr = atOp->getAttrOfType<mlir::IntegerAttr>("numBytes")) {
            auto numBytes = numBytesAttr.getInt();
            auto bits = numBytes * 8;
            if (bits == convertedType.getIntOrFloatBitWidth()) {
               //simple case: matches length of
               val = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(baseType), valueBuffer, index);
               val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
            } else {
               Value factor = rewriter.create<mlir::arith::ConstantIndexOp>(loc, numBytes);
               Value pos = rewriter.create<arith::AddIOp>(loc, columnOffset, index);
               pos = rewriter.create<arith::MulIOp>(loc, pos, factor);
               Value valBuffer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI8Type()), originalValueBuffer);
               Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos);
               auto combine = [loc](mlir::IntegerType largerType, mlir::Value first, mlir::Value second, size_t shiftAmount, mlir::OpBuilder& rewriter) -> mlir::Value {
                  mlir::Value ext1 = rewriter.create<arith::ExtUIOp>(loc, largerType, first);
                  mlir::Value ext2 = rewriter.create<arith::ExtUIOp>(loc, largerType, second);
                  mlir::Value shiftAmountConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, shiftAmount, largerType);
                  mlir::Value shifted = rewriter.create<mlir::arith::ShLIOp>(loc, ext2, shiftAmountConst);
                  return rewriter.create<mlir::arith::OrIOp>(loc, ext1, shifted);
               };
               if (bits == 24) {
                  mlir::Value i16Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI16Type()), ptr);
                  mlir::Value i16Val = rewriter.create<util::LoadOp>(loc, i16Ptr);
                  Value const2 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 2);
                  Value pos2 = rewriter.create<arith::AddIOp>(loc, pos, const2);
                  Value i8Ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos2);
                  mlir::Value i8Val = rewriter.create<util::LoadOp>(loc, i8Ptr);
                  val = combine(rewriter.getI32Type(), i16Val, i8Val, 16, rewriter);

                  i16Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
                  i8Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());

               } else if (bits == 40) {
                  mlir::Value i32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr);
                  mlir::Value i32Val = rewriter.create<util::LoadOp>(loc, i32Ptr);
                  Value const4 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 4);
                  Value pos2 = rewriter.create<arith::AddIOp>(loc, pos, const4);
                  Value i8Ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos2);
                  mlir::Value i8Val = rewriter.create<util::LoadOp>(loc, i8Ptr);
                  val = combine(rewriter.getI64Type(), i32Val, i8Val, 32, rewriter);

                  i32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
                  i8Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
               } else if (bits == 48) {
                  mlir::Value i32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr);
                  mlir::Value i32Val = rewriter.create<util::LoadOp>(loc, i32Ptr);
                  Value const4 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 4);
                  Value pos2 = rewriter.create<arith::AddIOp>(loc, pos, const4);
                  Value i8Ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos2);
                  mlir::Value i16Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI16Type()), i8Ptr);

                  mlir::Value i16Val = rewriter.create<util::LoadOp>(loc, i16Ptr);
                  val = combine(rewriter.getI64Type(), i32Val, i16Val, 32, rewriter);

                  i32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
                  i16Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
               } else if (bits == 56) {
                  mlir::Value i32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr);
                  mlir::Value i32Val = rewriter.create<util::LoadOp>(loc, i32Ptr);
                  Value const3 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 3);
                  Value const8i32 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 8, 32);

                  Value pos3 = rewriter.create<arith::AddIOp>(loc, pos, const3);
                  Value ptr3 = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos3);
                  mlir::Value secondI32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr3);

                  mlir::Value secondI32Val = rewriter.create<util::LoadOp>(loc, secondI32Ptr);
                  mlir::Value noDuplicate = rewriter.create<mlir::arith::ShRUIOp>(loc, secondI32Val, const8i32);
                  val = combine(rewriter.getI64Type(), i32Val, noDuplicate, 32, rewriter);

                  i32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
                  secondI32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
               }
            }
         } else {
            val = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(baseType), valueBuffer, index);
            val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         }
      }
      if (baseType.isa<util::RefType>()) { 
         auto numBytesAttr = atOp->getAttrOfType<mlir::IntegerAttr>("numBytes");
         Value vecSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, numBytesAttr.getInt());
         Value pos = rewriter.create<arith::AddIOp>(loc, columnOffset, index);
         pos = rewriter.create<arith::MulIOp>(loc, pos, vecSize);
         Value valBuffer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getF32Type()), originalValueBuffer);
         val = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getF32Type()), valBuffer, pos);
      } else {
         assert(val && "unhandled type!!");
      }
      if (atOp->getNumResults() == 2) {
         Value realPos = rewriter.create<arith::AddIOp>(loc, indexType, columnOffset, index);
         realPos = rewriter.create<arith::MulIOp>(loc, indexType, nullMultiplier, realPos);
         Value isValid = getBit(rewriter, loc, validityBuffer, realPos);
         rewriter.replaceOp(atOp, mlir::ValueRange{val, isValid});
      } else {
         rewriter.replaceOp(atOp, val);
      }
      return success();
   }
};
} // namespace

void mlir::dsa::populateCollectionsToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   auto* context = patterns.getContext();

   patterns.insert<ForOpLowering>(typeConverter, context);
   patterns.insert<AtLowering>(typeConverter, context);

   auto indexType = IndexType::get(context);
   auto i8ptrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));

   typeConverter.addConversion([context, i8ptrType, indexType](mlir::dsa::RecordBatchType recordBatchType) {
      std::vector<Type> types;
      types.push_back(indexType);
      if (auto tupleT = recordBatchType.getRowType().dyn_cast_or_null<TupleType>()) {
         for (auto t : tupleT.getTypes()) {
            if (t.isa<mlir::util::VarLen32Type>()) {
               t = mlir::IntegerType::get(context, 32);
            } else if (t == mlir::IntegerType::get(context, 1)) {
               t = mlir::IntegerType::get(context, 8);
            }

            types.push_back(indexType);
            types.push_back(indexType);
            types.push_back(i8ptrType);
            types.push_back(mlir::util::RefType::get(context, t));
            types.push_back(i8ptrType);
         }
      }
      return (Type) TupleType::get(context, types);
   });
   typeConverter.addConversion([context, &typeConverter, indexType](mlir::dsa::RecordType recordType) {
      return (Type) TupleType::get(context, {indexType, typeConverter.convertType(mlir::dsa::RecordBatchType::get(context, recordType.getRowType()))});
   });
}