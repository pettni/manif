#include <gtest/gtest.h>

#include "manif/Composite.h"
#include "manif/SO3.h"
#include "manif/SE3.h"
#include "manif/Rn.h"

#include "../common_tester.h"

#include <Eigen/StdVector>

using namespace manif;

using GroupT = Composite<double, R2, SO3, R1>;
using TangentT = CompositeTangent<double, R2, SO3, R1>;

TEST(Composite, Static)
{
  static_assert(internal::traits<GroupT>::Dim == 6);
  static_assert(internal::traits<GroupT>::DoF == 6);
  static_assert(internal::traits<GroupT>::RepSize == 7);

  static_assert(GroupT::DoF == R2d::DoF + SO3d::DoF + R1d::DoF);
  static_assert(GroupT::RepSize == R2d::RepSize + SO3d::RepSize + R1d::RepSize);
  static_assert(GroupT::Dim == R2d::Dim + SO3d::Dim + R1d::Dim);

  static_assert(
    GroupT::Vector::RowsAtCompileTime ==
    R2d::Vector::RowsAtCompileTime + SO3d::Vector::RowsAtCompileTime +
    R1d::Vector::RowsAtCompileTime);

  static_assert(
    GroupT::Transformation::RowsAtCompileTime ==
    R2d::Transformation::RowsAtCompileTime + SO3d::Transformation::RowsAtCompileTime +
    R1d::Transformation::RowsAtCompileTime);

  static_assert(
    GroupT::Transformation::ColsAtCompileTime ==
    R2d::Transformation::ColsAtCompileTime + SO3d::Transformation::ColsAtCompileTime +
    R1d::Transformation::ColsAtCompileTime);

  static_assert(GroupT::Tangent::Dim == R2d::Tangent::Dim + SO3d::Tangent::Dim + R1d::Tangent::Dim);
  static_assert(GroupT::Tangent::DoF == R2d::Tangent::DoF + SO3d::Tangent::DoF + R1d::Tangent::DoF);
  static_assert(GroupT::Tangent::RepSize
    == R2d::Tangent::RepSize + SO3d::Tangent::RepSize + R1d::Tangent::RepSize);

  static_assert(
    GroupT::Tangent::LieAlg::RowsAtCompileTime ==
    R2d::Tangent::LieAlg::RowsAtCompileTime + SO3d::Tangent::LieAlg::RowsAtCompileTime +
    R1d::Tangent::LieAlg::RowsAtCompileTime);

  static_assert(
    GroupT::Tangent::LieAlg::ColsAtCompileTime ==
    R2d::Tangent::LieAlg::ColsAtCompileTime + SO3d::Tangent::LieAlg::ColsAtCompileTime +
    R1d::Tangent::LieAlg::ColsAtCompileTime);

  static_assert(
    GroupT::Jacobian::RowsAtCompileTime ==
    R2d::Jacobian::RowsAtCompileTime + SO3d::Jacobian::RowsAtCompileTime +
    R1d::Jacobian::RowsAtCompileTime);

  static_assert(
    GroupT::Jacobian::ColsAtCompileTime ==
    R2d::Jacobian::ColsAtCompileTime + SO3d::Jacobian::ColsAtCompileTime +
    R1d::Jacobian::ColsAtCompileTime);

  static_assert(
    GroupT::Tangent::Jacobian::RowsAtCompileTime ==
    R2d::Tangent::Jacobian::RowsAtCompileTime + SO3d::Tangent::Jacobian::RowsAtCompileTime +
    R1d::Tangent::Jacobian::RowsAtCompileTime);

  static_assert(
    GroupT::Tangent::Jacobian::ColsAtCompileTime ==
    R2d::Tangent::Jacobian::ColsAtCompileTime + SO3d::Tangent::Jacobian::ColsAtCompileTime +
    R1d::Tangent::Jacobian::ColsAtCompileTime);
}


TEST(Composite, Intface)
{
  GroupT G(
    R2d(Eigen::Vector2d{1, 2}),
    SO3d{1., 2., 3.},
    R1d(Eigen::Matrix<double, 1, 1>{3})
  );


  auto Glog = G.log();
  EXPECT_EIGEN_NEAR(Glog.get<0>().coeffs(), G.get<0>().log().coeffs());
  EXPECT_EIGEN_NEAR(Glog.get<1>().coeffs(), G.get<1>().log().coeffs());
  EXPECT_EIGEN_NEAR(Glog.get<2>().coeffs(), G.get<2>().log().coeffs());

  auto Ginv = G.inverse();
  EXPECT_EIGEN_NEAR(G.get<0>().inverse().coeffs(), Ginv.get<0>().coeffs());
  EXPECT_EIGEN_NEAR(G.get<1>().inverse().coeffs(), Ginv.get<1>().coeffs());
  EXPECT_EIGEN_NEAR(G.get<2>().inverse().coeffs(), Ginv.get<2>().coeffs());

  auto G_Ginv = G.compose(Ginv);
  EXPECT_EIGEN_NEAR(G_Ginv.get<0>().inverse().coeffs(), R2d::Identity().coeffs());
  EXPECT_EIGEN_NEAR(G_Ginv.get<1>().inverse().coeffs(), SO3d::Identity().coeffs());
  EXPECT_EIGEN_NEAR(G_Ginv.get<2>().inverse().coeffs(), R1d::Identity().coeffs());

  typename GroupT::Vector vec;

  auto adj = G.adj();
  Eigen::Matrix2d adj0 = adj.block<2, 2>(0, 0);
  EXPECT_EIGEN_NEAR(adj0, G.get<0>().adj());
  Eigen::Matrix3d adj1 = adj.block<3, 3>(2, 2);
  EXPECT_EIGEN_NEAR(adj1, G.get<1>().adj());
  Eigen::Matrix<double, 1, 1> adj2 = adj.block<1, 1>(5, 5);
  EXPECT_EIGEN_NEAR(adj2, G.get<2>().adj());
}


TEST(Composite, Map)
{
  std::array<double, GroupT::RepSize> data;
  data.fill(1);

  Eigen::Map<GroupT> map(data.data());
}


TEST(CompositeTangent, Interface)
{
  TangentT tangent;

  tangent.get<0>() = Eigen::Vector2d{1, 2};
  tangent.get<1>() = Eigen::Vector3d{3, 4, 5};
  tangent.get<2>() = Eigen::Matrix<double, 1, 1>{6};

  auto exp = tangent.exp();

  EXPECT_EIGEN_NEAR(exp.get<0>().coeffs(), tangent.get<0>().exp().coeffs());
  EXPECT_EIGEN_NEAR(exp.get<1>().coeffs(), tangent.get<1>().exp().coeffs());
  EXPECT_EIGEN_NEAR(exp.get<2>().coeffs(), tangent.get<2>().exp().coeffs());
}


TEST(CompositeTangent, Jacobians)
{
  GroupT G(
    R2d(Eigen::Vector2d{1, 2}),
    SO3d{1., 2., 3.},
    R1d(Eigen::Matrix<double, 1, 1>{3})
  );
  auto tangent = G.log();

  {
    auto jac = tangent.rjac();
    Eigen::Matrix2d jac0 = jac.block<2, 2>(0, 0);
    Eigen::Matrix3d jac1 = jac.block<3, 3>(2, 2);
    Eigen::Matrix<double, 1, 1> jac2 = jac.block<1, 1>(5, 5);

    EXPECT_EIGEN_NEAR(jac0, tangent.get<0>().rjac());
    EXPECT_EIGEN_NEAR(jac1, tangent.get<1>().rjac());
    EXPECT_EIGEN_NEAR(jac2, tangent.get<2>().rjac());
  }

  {
    auto jac = tangent.ljac();
    Eigen::Matrix2d jac0 = jac.block<2, 2>(0, 0);
    Eigen::Matrix3d jac1 = jac.block<3, 3>(2, 2);
    Eigen::Matrix<double, 1, 1> jac2 = jac.block<1, 1>(5, 5);

    EXPECT_EIGEN_NEAR(jac0, tangent.get<0>().ljac());
    EXPECT_EIGEN_NEAR(jac1, tangent.get<1>().ljac());
    EXPECT_EIGEN_NEAR(jac2, tangent.get<2>().ljac());
  }

  {
    auto jac = tangent.rjacinv();
    Eigen::Matrix2d jac0 = jac.block<2, 2>(0, 0);
    Eigen::Matrix3d jac1 = jac.block<3, 3>(2, 2);
    Eigen::Matrix<double, 1, 1> jac2 = jac.block<1, 1>(5, 5);

    EXPECT_EIGEN_NEAR(jac0, tangent.get<0>().rjacinv());
    EXPECT_EIGEN_NEAR(jac1, tangent.get<1>().rjacinv());
    EXPECT_EIGEN_NEAR(jac2, tangent.get<2>().rjacinv());
  }


  {
    auto jac = tangent.ljacinv();
    Eigen::Matrix2d jac0 = jac.block<2, 2>(0, 0);
    Eigen::Matrix3d jac1 = jac.block<3, 3>(2, 2);
    Eigen::Matrix<double, 1, 1> jac2 = jac.block<1, 1>(5, 5);

    EXPECT_EIGEN_NEAR(jac0, tangent.get<0>().ljacinv());
    EXPECT_EIGEN_NEAR(jac1, tangent.get<1>().ljacinv());
    EXPECT_EIGEN_NEAR(jac2, tangent.get<2>().ljacinv());
  }

  {
    auto jac = tangent.smallAdj();
    Eigen::Matrix2d jac0 = jac.block<2, 2>(0, 0);
    Eigen::Matrix3d jac1 = jac.block<3, 3>(2, 2);
    Eigen::Matrix<double, 1, 1> jac2 = jac.block<1, 1>(5, 5);

    EXPECT_EIGEN_NEAR(jac0, tangent.get<0>().smallAdj());
    EXPECT_EIGEN_NEAR(jac1, tangent.get<1>().smallAdj());
    EXPECT_EIGEN_NEAR(jac2, tangent.get<2>().smallAdj());
  }
}


MANIF_TEST(GroupT);

MANIF_TEST_MAP(GroupT);

MANIF_TEST_JACOBIANS(GroupT);


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
