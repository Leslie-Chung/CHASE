#ifndef DBInterfaces
#define DBInterfaces
include "mlir/IR/OpBase.td"

def DB_NeedsNullWrap : OpInterface<"NeedsNullWrap"> {
    let description = [{
        This interface allows operations to indicate that they cannot handle
        `NULL` values themselves and need a wrapper.
    }];
    let cppNamespace = "::mlir::db";

    let methods = [
        InterfaceMethod<"Returns true if the operator cannot `NULL` itself.",
            "bool", "needsNullWrap", (ins), [{}],
            /*defaultImplementation=*/[{
                return true;
            }]
        >,
    ];
}

/*
class DeclareOpInterfaceMethods<OpInterface interface,
                                list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        OpInterface<interface.cppInterfaceName, interface.baseInterfaces> {
    let description = interface.description;
    let cppInterfaceName = interface.cppInterfaceName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
    let baseInterfaces = interface.baseInterfaces;
}
*/
def DB_MayNeedNullWrap : DeclareOpInterfaceMethods<DB_NeedsNullWrap,["needsNullWrap"]> {}

def DB_SupportsInvalidValues : OpInterface<"SupportsInvalidValues"> {
    let description = [{
        This interface allows operations to indicate that they can safely handle
        invalid (garbage) values (and therefore can be safely executed on the
        invalid value storage of `NULL` values).
    }];
    let cppNamespace = "::mlir::db";

    let methods = [
        InterfaceMethod<"Returns true if the operator supports invalid values.",
            "bool", "supportsInvalidValues", (ins), [{}],
            /*defaultImplementation=*/[{
                return true;
            }]
        >,
    ];
}

def DB_MaySupportInvalidValues : DeclareOpInterfaceMethods<DB_SupportsInvalidValues,["supportsInvalidValues"]> {}

#endif// DBInterfaces
