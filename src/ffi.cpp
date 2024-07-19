#include <llvm-c/Support.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

using namespace mlir::LLVM;
using namespace mlir;

extern "C" const void *LLVMStructType_getFieldTypeAt(const void *ty_ptr,
                                                     unsigned index) {
  mlir::LLVM::LLVMStructType type =
      mlir::LLVM::LLVMStructType::getFromOpaquePointer(ty_ptr);

  return type.getBody()[index].getAsOpaquePointer();
}

extern "C" LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirOperation module,
                                                     LLVMContextRef context) {
  mlir::Operation *moduleOp = unwrap(module);

  llvm::LLVMContext *ctx = llvm::unwrap(context);

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(moduleOp, *ctx);

  LLVMModuleRef moduleRef = llvm::wrap(llvmModule.release());

  return moduleRef;
}

enum MlirLLVMTypeEncoding {
  MlirLLVMTypeEncodingAddress = 0x1,
  MlirLLVMTypeEncodingBoolean = 0x2,
  MlirLLVMTypeEncodingComplexFloat = 0x31,
  MlirLLVMTypeEncodingFloatT = 0x4,
  MlirLLVMTypeEncodingSigned = 0x5,
  MlirLLVMTypeEncodingSignedChar = 0x6,
  MlirLLVMTypeEncodingUnsigned = 0x7,
  MlirLLVMTypeEncodingUnsignedChar = 0x08,
  MlirLLVMTypeEncodingImaginaryFloat = 0x09,
  MlirLLVMTypeEncodingPackedDecimal = 0x0a,
  MlirLLVMTypeEncodingNumericString = 0x0b,
  MlirLLVMTypeEncodingEdited = 0x0c,
  MlirLLVMTypeEncodingSignedFixed = 0x0d,
  MlirLLVMTypeEncodingUnsignedFixed = 0x0e,
  MlirLLVMTypeEncodingDecimalFloat = 0x0f,
  MlirLLVMTypeEncodingUTF = 0x10,
  MlirLLVMTypeEncodingUCS = 0x11,
  MlirLLVMTypeEncodingASCII = 0x12,
  MlirLLVMTypeEncodingLoUser = 0x80,
  MlirLLVMTypeEncodingHiUser = 0xff,
};
typedef enum MlirLLVMTypeEncoding MlirLLVMTypeEncoding;

extern "C" MlirAttribute
mlirLLVMDIBasicTypeAttrGet(MlirContext ctx, unsigned int tag,
                           MlirAttribute name, uint64_t sizeInBits,
                           MlirLLVMTypeEncoding encoding) {

  return wrap(DIBasicTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)), sizeInBits, encoding));
}

enum MlirLLVMDIFlags {
  MlirLLVMDIFlagsZero = 0,
  MlirLLVMDIFlagsBit0 = 1,
  MlirLLVMDIFlagsBit1 = 2,
  MlirLLVMDIFlagsPrivate = 1,
  MlirLLVMDIFlagsProtected = 2,
  MlirLLVMDIFlagsPublic = 3,
  MlirLLVMDIFlagsFwdDecl = 4,
  MlirLLVMDIFlagsAppleBlock = 8,
  MlirLLVMDIFlagsReservedBit4 = 16,
  MlirLLVMDIFlagsVirtual = 32,
  MlirLLVMDIFlagsArtificial = 64,
  MlirLLVMDIFlagsExplicit = 128,
  MlirLLVMDIFlagsPrototyped = 256,
  MlirLLVMDIFlagsObjcClassComplete = 512,
  MlirLLVMDIFlagsObjectPointer = 1024,
  MlirLLVMDIFlagsVector = 2048,
  MlirLLVMDIFlagsStaticMember = 4096,
  MlirLLVMDIFlagsLValueReference = 8192,
  MlirLLVMDIFlagsRValueReference = 16384,
  MlirLLVMDIFlagsExportSymbols = 32768,
  MlirLLVMDIFlagsSingleInheritance = 65536,
  MlirLLVMDIFlagsMultipleInheritance = 65536,
  MlirLLVMDIFlagsVirtualInheritance = 65536,
  MlirLLVMDIFlagsIntroducedVirtual = 262144,
  MlirLLVMDIFlagsBitField = 524288,
  MlirLLVMDIFlagsNoReturn = 1048576,
  MlirLLVMDIFlagsTypePassByValue = 4194304,
  MlirLLVMDIFlagsTypePassByReference = 8388608,
  MlirLLVMDIFlagsEnumClass = 16777216,
  MlirLLVMDIFlagsThunk = 33554432,
  MlirLLVMDIFlagsNonTrivial = 67108864,
  MlirLLVMDIFlagsBigEndian = 134217728,
  MlirLLVMDIFlagsLittleEndian = 268435456,
  MlirLLVMDIFlagsAllCallsDescribed = 536870912,
};
typedef enum MlirLLVMDIFlags MlirLLVMDIFlags;

extern "C" MlirAttribute mlirLLVMDINullTypeAttrGet(MlirContext ctx) {
  return wrap(DINullTypeAttr::get(unwrap(ctx)));
}

extern "C" MlirAttribute mlirLLVMDIFileAttrGet(MlirContext ctx,
                                               MlirAttribute name,
                                               MlirAttribute directory) {
  return wrap(DIFileAttr::get(unwrap(ctx), cast<StringAttr>(unwrap(name)),
                              cast<StringAttr>(unwrap(directory))));
}

enum MlirLLVMDIEmissionKind {
  MlirLLVMDIEmissionKindNone = 0,
  MlirLLVMDIEmissionKindFull = 1,
  MlirLLVMDIEmissionKindLineTablesOnly = 2,
  MlirLLVMDIEmissionKindDebugDirectivesOnly = 3,
};
typedef enum MlirLLVMDIEmissionKind MlirLLVMDIEmissionKind;

enum MlirLLVMDINameTableKind {
  MlirLLVMDINameTableKindDefault = 0,
  MlirLLVMDINameTableKindGNU = 1,
  MlirLLVMDINameTableKindNone = 2,
  MlirLLVMDINameTableKindApple = 3,
};
typedef enum MlirLLVMDINameTableKind MlirLLVMDINameTableKind;

enum MlirLLVMDISubprogramFlags {
  MlirLLVMDISubprogramFlagsVirtual = 1,
  MlirLLVMDISubprogramFlagsPureVirtual = 2,
  MlirLLVMDISubprogramFlagsLocalToUnit = 4,
  MlirLLVMDISubprogramFlagsDefinition = 8,
  MlirLLVMDISubprogramFlagsOptimized = 16,
  MlirLLVMDISubprogramFlagsPure = 32,
  MlirLLVMDISubprogramFlagsElemental = 64,
  MlirLLVMDISubprogramFlagsRecursive = 128,
  MlirLLVMDISubprogramFlagsMainSubprogram = 256,
  MlirLLVMDISubprogramFlagsDeleted = 512,
  MlirLLVMDISubprogramFlagsObjCDirect = 2048,
};
typedef enum MlirLLVMDISubprogramFlags MlirLLVMDISubprogramFlags;

extern "C" MlirAttribute
mlirLLVMDistinctAttrCreate(MlirAttribute referenced_attr) {
  return wrap(DistinctAttr::create(unwrap(referenced_attr)));
}

extern "C" MlirAttribute
mlirLLVMDICompileUnitAttrGet(MlirContext ctx, MlirAttribute id,
                             unsigned int sourceLanguage, MlirAttribute file,
                             MlirAttribute producer, bool isOptimized,
                             MlirLLVMDIEmissionKind emissionKind) {
  return wrap(DICompileUnitAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)), sourceLanguage,
      cast<DIFileAttr>(unwrap(file)), cast<StringAttr>(unwrap(producer)),
      isOptimized, DIEmissionKind(emissionKind)));
}

extern "C" MlirAttribute
mlirLLVMDICompileUnitAttrGetScope(MlirContext ctx, MlirAttribute cu) {
  return wrap(cast<DICompileUnitAttr>(unwrap(cu)).getFile());
}

extern "C" MlirAttribute mlirLLVMDIFlagsAttrGet(MlirContext ctx,
                                                MlirLLVMDIFlags value) {
  return wrap(DIFlagsAttr::get(unwrap(ctx), DIFlags(value)));
}

extern "C" MlirAttribute mlirLLVMDILexicalBlockAttrGet(MlirContext ctx,
                                                       MlirAttribute scope,
                                                       MlirAttribute file,
                                                       unsigned int line,
                                                       unsigned int column) {
  return wrap(
      DILexicalBlockAttr::get(unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
                              cast<DIFileAttr>(unwrap(file)), line, column));
}

extern "C" MlirAttribute
mlirLLVMDISubroutineTypeAttrGet(MlirContext ctx, unsigned int callingConvention,
                                intptr_t nTypes, MlirAttribute const *types) {
  SmallVector<Attribute> attrStorage;
  attrStorage.reserve(nTypes);

  return wrap(DISubroutineTypeAttr::get(
      unwrap(ctx), callingConvention,
      llvm::map_to_vector(unwrapList(nTypes, types, attrStorage),
                          [](Attribute a) { return cast<DITypeAttr>(a); })));
}

extern "C" MlirAttribute mlirLLVMDISubprogramAttrGet(
    MlirContext ctx, MlirAttribute id, MlirAttribute compileUnit,
    MlirAttribute scope, MlirAttribute name, MlirAttribute linkageName,
    MlirAttribute file, unsigned int line, unsigned int scopeLine,
    MlirLLVMDISubprogramFlags subprogramFlags, MlirAttribute type) {
  return wrap(DISubprogramAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)),
      cast<DICompileUnitAttr>(unwrap(compileUnit)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(linkageName)), cast<DIFileAttr>(unwrap(file)),
      line, scopeLine, DISubprogramFlags(subprogramFlags),
      cast<DISubroutineTypeAttr>(unwrap(type))));
}

extern "C" MlirAttribute
mlirLLVMDISubprogramAttrGetScope(MlirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getScope());
}

extern "C" MlirAttribute mlirLLVMDIModuleAttrGet(MlirContext ctx, MlirAttribute file,
                                      MlirAttribute scope, MlirAttribute name,
                                      MlirAttribute configMacros,
                                      MlirAttribute includePath,
                                      MlirAttribute apinotes, unsigned int line,
                                      bool isDecl) {
  return wrap(DIModuleAttr::get(
      unwrap(ctx), cast<DIFileAttr>(unwrap(file)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(configMacros)),
      cast<StringAttr>(unwrap(includePath)), cast<StringAttr>(unwrap(apinotes)),
      line, isDecl));
}

extern "C" MlirAttribute mlirLLVMDIModuleAttrGetScope(MlirAttribute diModule) {
  return wrap(cast<DIModuleAttr>(unwrap(diModule)).getScope());
}
