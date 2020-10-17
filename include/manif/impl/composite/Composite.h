#ifndef _MANIF_MANIF_COMPOSITE_H_
#define _MANIF_MANIF_COMPOSITE_H_

#include "manif/impl/composite/Composite_base.h"

#include <tuple>
#include <utility>

namespace manif
{
namespace internal
{

//! Traits specialization
template<typename _Scalar, template<typename> typename ... _T>
struct traits<Composite<_Scalar, _T ...>>
{
  // Composite-specific traits
  using IdxList = m_make_intseq<sizeof...(_T)>;

  using LenDim = m_intseq<_T<_Scalar>::Dim ...>;
  using BegDim = composite::intseq_psum_t<LenDim>;

  using LenDoF = m_intseq<_T<_Scalar>::DoF ...>;
  using BegDoF = composite::intseq_psum_t<LenDoF>;

  using LenTra = m_intseq<_T<_Scalar>::Transformation::RowsAtCompileTime ...>;
  using BegTra = composite::intseq_psum_t<LenTra>;

  using LenRep = m_intseq<_T<_Scalar>::RepSize ...>;
  using BegRep = composite::intseq_psum_t<LenRep>;

  template<size_t _Idx>
  using PartType = typename std::tuple_element<_Idx, std::tuple<_T<_Scalar>...>>::type;

  // Regular traits
  using Scalar = _Scalar;

  using LieGroup = Composite<_Scalar, _T ...>;
  using Tangent = CompositeTangent<_Scalar, _T ...>;

  using Base = CompositeBase<Composite<_Scalar, _T ...>>;

  static constexpr int Dim = composite::intseq_sum<LenDim>::value;
  static constexpr int DoF = composite::intseq_sum<LenDoF>::value;
  static constexpr int RepSize = composite::intseq_sum<LenRep>::value;

  using DataType = Eigen::Matrix<_Scalar, RepSize, 1>;
  using Jacobian = Eigen::Matrix<_Scalar, DoF, DoF>;
  using Transformation = Eigen::Matrix<
    _Scalar, composite::intseq_sum<LenTra>::value, composite::intseq_sum<LenTra>::value
  >;
  using Vector = Eigen::Matrix<_Scalar, Dim, 1>;
};

}  // namespace internal

//
// Composite LieGroup
//

/**
 * @brief Represents a composite element.
 */
template<typename _Scalar, template<typename> typename ... _T>
struct Composite : CompositeBase<Composite<_Scalar, _T ...>>
{
private:
  static_assert(sizeof...(_T) > 0, "Must have at least one element in composite !");

  using Base = CompositeBase<Composite<_Scalar, _T...>>;
  using Type = Composite<_Scalar, _T...>;

  using BegRep = typename internal::traits<Composite>::BegRep;
  using LenRep = typename internal::traits<Composite>::LenRep;

public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND

  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Composite()  = default;
  ~Composite() = default;

  // Copy constructor given base
  Composite(const Base & o);

  template<typename _DerivedOther>
  Composite(const CompositeBase<_DerivedOther> & o);

  template<typename _DerivedOther>
  Composite(const LieGroupBase<_DerivedOther> & o);

  // Copy constructor given Eigen
  template<typename _EigenDerived>
  Composite(const Eigen::MatrixBase<_EigenDerived> & data);

  // LieGroup common API

  //! Get a const reference to the underlying DataType.
  DataType & coeffs();

  //! Get a const reference to the underlying DataType.
  const DataType & coeffs() const;

  // Composite specific API

  //! Construct from individual parts
  Composite(_T<_Scalar> && ... parts);

protected:
  // Helper for the parts constructor
  template<int ... _BegRep, int ... _LenRep>
  Composite(m_intseq<_BegRep...>, m_intseq<_LenRep...>, _T<_Scalar> && ... parts);

protected:
  DataType data_;
};

MANIF_EXTRA_GROUP_TYPEDEF(Composite)

template<typename _Scalar, template<typename> typename ... _T>
Composite<_Scalar, _T...>::Composite(const Base & o)
: Composite(o.coeffs())
{}

template<typename _Scalar, template<typename> typename ... _T>
template<typename _DerivedOther>
Composite<_Scalar, _T...>::Composite(const CompositeBase<_DerivedOther> & o)
: Composite(o.coeffs())
{}

template<typename _Scalar, template<typename> typename ... _T>
template<typename _DerivedOther>
Composite<_Scalar, _T...>::Composite(const LieGroupBase<_DerivedOther> & o)
: Composite(o.coeffs())
{}

template<typename _Scalar, template<typename> typename ... _T>
template<typename _EigenDerived>
Composite<_Scalar, _T...>::Composite(const Eigen::MatrixBase<_EigenDerived> & data)
: data_(data)
{}

template<typename _Scalar, template<typename> typename ... _T>
Composite<_Scalar, _T...>::Composite(_T<_Scalar> && ... parts)
: Composite(BegRep{}, LenRep{}, std::forward<_T<_Scalar>>(parts) ...)
{}

template<typename _Scalar, template<typename> typename ... _T>
template<int ... _BegRep, int ... _LenRep>
Composite<_Scalar, _T...>::Composite(
  m_intseq<_BegRep...>, m_intseq<_LenRep...>, _T<_Scalar> && ... parts)
{
  // c++11 "fold expression"
  auto l = {((data_.template segment<_LenRep>(_BegRep) =
    std::forward<_T<_Scalar>>(parts).coeffs()), 0) ...};
  static_cast<void>(l);  // compiler warning
}

template<typename _Scalar, template<typename> typename ... _T>
typename Composite<_Scalar, _T...>::DataType &
Composite<_Scalar, _T...>::coeffs()
{
  return data_;
}

template<typename _Scalar, template<typename> typename ... _T>
const typename Composite<_Scalar, _T...>::DataType &
Composite<_Scalar, _T...>::coeffs() const
{
  return data_;
}

} // namespace manif

#endif // _MANIF_MANIF_COMPOSITE_H_
