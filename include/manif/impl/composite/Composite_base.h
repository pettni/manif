#ifndef _MANIF_MANIF_COMPOSITE_BASE_H_
#define _MANIF_MANIF_COMPOSITE_BASE_H_

#include "manif/impl/composite/Composite_properties.h"
#include "manif/impl/lie_group_base.h"

#include <iostream>

namespace manif
{

/**
 * @brief The base class of the Composite group.
 */
template<typename _Derived>
struct CompositeBase : LieGroupBase<_Derived>
{
private:
  using Base = LieGroupBase<_Derived>;
  using Type = CompositeBase<_Derived>;

  using IdxList = typename internal::traits<_Derived>::IdxList;

  using BegDoF = typename internal::traits<_Derived>::BegDoF;
  using LenDoF = typename internal::traits<_Derived>::LenDoF;

  using BegDim = typename internal::traits<_Derived>::BegDim;
  using LenDim = typename internal::traits<_Derived>::LenDim;

  using BegRep = typename internal::traits<_Derived>::BegRep;
  using LenRep = typename internal::traits<_Derived>::LenRep;

  using BegTra = typename internal::traits<_Derived>::BegTra;
  using LenTra = typename internal::traits<_Derived>::LenTra;

  template<int Idx>
  using PartType = typename internal::traits<_Derived>::template PartType<Idx>;

public:
  MANIF_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_AUTO_API
  MANIF_INHERIT_GROUP_OPERATOR

  using Base::coeffs;

  using Transformation = typename internal::traits<_Derived>::Transformation;


  // LieGroup common API

  /**
   * @brief Get the inverse of this.
   * @param[out] -optional- J_minv_m Jacobian of the inverse wrt this.
   * @note r^-1 = -r
   */
  LieGroup inverse(OptJacobianRef J_minv_m = {}) const;

  /**
   * @brief Get the corresponding Lie algebra element in vector form.
   * @param[out] -optional- J_t_m Jacobian of the tangent wrt to this.
   * @return The tangent of this.
   * @note This is the log() map in vector form.
   */
  Tangent log(OptJacobianRef J_t_m = {}) const;

  /**
   * @brief This function is deprecated.
   * Please considere using
   * @ref log instead.
   */
  MANIF_DEPRECATED
  Tangent lift(OptJacobianRef J_t_m = {}) const;

  /**
   * @brief Composition of this and another Rn element.
   * @param[in] m Another Rn element.
   * @param[out] -optional- J_mc_ma Jacobian of the composition wrt this.
   * @param[out] -optional- J_mc_mb Jacobian of the composition wrt m.
   * @return The composition of 'this . m'.
   * @note See Eq. (190).
   */
  template<typename _DerivedOther>
  LieGroup compose(
    const LieGroupBase<_DerivedOther> & m,
    OptJacobianRef J_mc_ma = {},
    OptJacobianRef J_mc_mb = {}) const;

  /**
   * @brief Composite group action
   * @param v vector.
   * @param[out] -optional- J_vout_m The Jacobian of the new object wrt this.
   * @param[out] -optional- J_vout_v The Jacobian of the new object wrt input object.
   * @return The translated vector.
   */
  Vector act(
    const Vector & v,
    tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> J_vout_m = {},
    tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> J_vout_v = {}) const;

  /**
   * @brief Get the ajoint matrix at this.
   */
  Jacobian adj() const;


  // Composite-specific API

  /**
   * @brief Get the block-diagonal composite transformation matrix
   */
  Transformation transform() const;

  //! Get a map to a part of the composite
  template<int _Idx>
  Eigen::Map<PartType<_Idx>> get();

  //! Get a const map to a part of the composite
  template<int _Idx>
  Eigen::Map<const PartType<_Idx>> get() const;

protected:
  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  LieGroup
    inverse_impl(OptJacobianRef, intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  Tangent
    log_impl(OptJacobianRef, intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const;

  template<typename _DerivedOther, int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  LieGroup
  compose_impl(
    const LieGroupBase<_DerivedOther> & m,
    OptJacobianRef J_mc_ma, OptJacobianRef J_mc_mb,
    intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDim, int ... _LenDim, int ... _BegDoF, int ... _LenDoF>
  Vector act_impl(
    const Vector & v,
    tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> J_vout_m,
    tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> J_vout_v,
    intseq<_Idx...>, intseq<_BegDim...>, intseq<_LenDim...>,
    intseq<_BegDoF...>, intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  Jacobian adj_impl(intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegTra, int ... _LenTra>
  Transformation
  transform_impl(intseq<_Idx...>, intseq<_BegTra...>, intseq<_LenTra...>) const;

  friend internal::RandomEvaluatorImpl<CompositeBase<_Derived>>;
};


template<typename _Derived>
typename CompositeBase<_Derived>::Transformation
CompositeBase<_Derived>::transform() const
{
  return transform_impl(IdxList{}, BegTra{}, LenTra{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegTra, int ... _LenTra>
typename CompositeBase<_Derived>::Transformation
CompositeBase<_Derived>::transform_impl(
  intseq<_Idx...>, intseq<_BegTra...>, intseq<_LenTra...>) const
{
  Transformation ret = Transformation::Zero();
  // cxx11 "fold expression"
  auto l =
  {((ret.template block<_LenTra, _LenTra>(_BegTra, _BegTra) = get<_Idx>().transform()), 0) ...};
  static_cast<void>(l);  // compiler warning
  return ret;
}

template<typename _Derived>
typename CompositeBase<_Derived>::LieGroup
CompositeBase<_Derived>::inverse(OptJacobianRef J_minv_m) const
{
  if (J_minv_m) {
    J_minv_m->setZero();
  }
  return inverse_impl(J_minv_m, IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeBase<_Derived>::LieGroup
CompositeBase<_Derived>::inverse_impl(
  OptJacobianRef J_minv_m, intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const
{
  if (J_minv_m) {
    return LieGroup(
      get<_Idx>().inverse(J_minv_m->template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF)) ...
    );
  }
  return LieGroup(get<_Idx>().inverse() ...);
}

template<typename _Derived>
typename CompositeBase<_Derived>::Tangent
CompositeBase<_Derived>::log(OptJacobianRef J_t_m) const
{
  if (J_t_m) {
    J_t_m->setZero();
  }
  return log_impl(J_t_m, IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeBase<_Derived>::Tangent
CompositeBase<_Derived>::log_impl(
  OptJacobianRef J_minv_m, intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const
{
  if (J_minv_m) {
    return Tangent(
      get<_Idx>().log(J_minv_m->template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF))...
    );
  }
  return Tangent(get<_Idx>().log() ...);
}

template<typename _Derived>
typename CompositeBase<_Derived>::Tangent
CompositeBase<_Derived>::lift(OptJacobianRef J_t_m) const
{
  return log(J_t_m);
}

template<typename _Derived>
template<typename _DerivedOther>
typename CompositeBase<_Derived>::LieGroup
CompositeBase<_Derived>::compose(
  const LieGroupBase<_DerivedOther> & m, OptJacobianRef J_mc_ma, OptJacobianRef J_mc_mb) const
{
  if (J_mc_ma) {
    J_mc_ma->setZero();
  }
  if (J_mc_mb) {
    J_mc_mb->setZero();
  }
  return compose_impl(m, J_mc_ma, J_mc_mb, IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
template<typename _DerivedOther, int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeBase<_Derived>::LieGroup
CompositeBase<_Derived>::compose_impl(
  const LieGroupBase<_DerivedOther> & m, OptJacobianRef J_mc_ma, OptJacobianRef J_mc_mb,
  intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const
{
  return LieGroup(
    get<_Idx>().compose(
      static_cast<const _DerivedOther &>(m).template get<_Idx>(),
      J_mc_ma ?
      J_mc_ma->template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) :
      tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _LenDoF, _LenDoF>>>{},
      J_mc_mb ?
      J_mc_mb->template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) :
      tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _LenDoF, _LenDoF>>>{}
    ) ...
  );
}

template<typename _Derived>
typename CompositeBase<_Derived>::Vector
CompositeBase<_Derived>::act(
  const typename CompositeBase<_Derived>::Vector & v,
  tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> J_vout_m,
  tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> J_vout_v) const
{
  if (J_vout_m) {
    J_vout_m->setZero();
  }
  if (J_vout_v) {
    J_vout_v->setZero();
  }

  return act_impl(v, J_vout_m, J_vout_v, IdxList{}, BegDim{}, LenDim{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDim, int ... _LenDim, int ... _BegDoF, int ... _LenDoF>
typename CompositeBase<_Derived>::Vector
CompositeBase<_Derived>::act_impl(
  const typename CompositeBase<_Derived>::Vector & v,
  tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, DoF>>> J_vout_m,
  tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, Dim, Dim>>> J_vout_v,
  intseq<_Idx...>, intseq<_BegDim...>, intseq<_LenDim...>,
  intseq<_BegDoF...>, intseq<_LenDoF...>) const
{
  Vector ret;
  // cxx11 "fold expression"
  auto l = {((ret.template segment<_LenDim>(_BegDim) = get<_Idx>().act(
      v.template segment<_LenDim>(_BegDim),
      J_vout_m ?
      J_vout_m->template block<_LenDim, _LenDoF>(_BegDim, _BegDoF) :
      tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _LenDim, _LenDoF>>>{},
      J_vout_v ?
      J_vout_v->template block<_LenDim, _LenDim>(_BegDim, _BegDim) :
      tl::optional<Eigen::Ref<Eigen::Matrix<Scalar, _LenDim, _LenDim>>>{}
    )), 0) ...};
  static_cast<void>(l);  // compiler warning
  return ret;
}

template<typename _Derived>
typename CompositeBase<_Derived>::Jacobian
CompositeBase<_Derived>::adj() const
{
  return adj_impl(IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeBase<_Derived>::Jacobian
CompositeBase<_Derived>::adj_impl(
  intseq<_Idx...>, intseq<_BegDoF...>, intseq<_LenDoF...>) const
{
  Jacobian adj = Jacobian::Zero();
  // cxx11 "fold expression"
  auto l = {((adj.template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) = get<_Idx>().adj()), 0) ...};
  static_cast<void>(l);  // compiler warning
  return adj;
}

template<typename _Derived>
template<int _Idx>
Eigen::Map<typename CompositeBase<_Derived>::template PartType<_Idx>>
CompositeBase<_Derived>::get()
{
  return Eigen::Map<PartType<_Idx>>(
    static_cast<_Derived &>(*this).coeffs().data() +
    internal::composite::intseq_element<_Idx, BegRep>::value);
}

template<typename _Derived>
template<int _Idx>
Eigen::Map<const typename CompositeBase<_Derived>::template PartType<_Idx>>
CompositeBase<_Derived>::get() const
{
  return Eigen::Map<const PartType<_Idx>>(
    static_cast<const _Derived &>(*this).coeffs().data() + internal::composite::intseq_element<_Idx,
    BegRep>::value);
}

namespace internal
{

//! @brief Random specialization for CompositeBase objects.
template<typename Derived>
struct RandomEvaluatorImpl<CompositeBase<Derived>>
{
  static void run(CompositeBase<Derived> & m)
  {
    run(m, typename CompositeBase<Derived>::IdxList{});
  }

  template<int ... _Idx>
  static void run(CompositeBase<Derived> & m, intseq<_Idx...>)
  {
    m = typename CompositeBase<Derived>::LieGroup(
      CompositeBase<Derived>::template PartType<_Idx>::Random() ...);
  }
};

}  // namespace internal
}  // namespace manif

#endif  // _MANIF_MANIF_COMPOSITE_BASE_H_
