# Exclude tests from libMLIR.so
add_mlir_library(MLIRLinalgTestPasses
  TestConvVectorization.cpp
  TestLinalgCodegenStrategy.cpp
  TestLinalgElementwiseFusion.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgHoisting.cpp
  TestLinalgTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRGPU
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRPass
  MLIRStandard
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToSCF
  )
