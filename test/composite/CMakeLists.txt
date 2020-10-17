# SO3 tests

manif_add_gtest(gtest_composite gtest_composite.cpp)

set(CXX_11_TEST_TARGETS

  ${CXX_11_TEST_TARGETS}

  # R^n
  gtest_composite

  PARENT_SCOPE
)
