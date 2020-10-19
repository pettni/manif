#ifndef _MANIF_MANIF_COMPOSITE_PROPERTIES_H_
#define _MANIF_MANIF_COMPOSITE_PROPERTIES_H_

#include "manif/impl/traits.h"

namespace manif
{

// Forward declaration for type traits specialization
template<typename _Derived>
struct CompositeBase;
template<typename _Derived>
struct CompositeTangentBase;

template<typename _Scalar, template<typename> class ... _T>
struct Composite;
template<typename _Scalar, template<typename> class ... _T>
struct CompositeTangent;

//! std::integer_sequence-equivalent
template<int ... _I>
struct intseq
{
  using value_type = int;
  static constexpr std::size_t size() noexcept {return sizeof...(_I);}
};

namespace internal
{
namespace composite
{

//! join two intseqs
template<typename _Seq1, typename _Seq2>
struct intseq_join;

template<template<int ...> class _IntSeq, int ... _I1, int ... _I2>
struct intseq_join<_IntSeq<_I1...>, _IntSeq<_I2...>>
{
  using type = _IntSeq<_I1..., _I2...>;
};

template<typename _Seq1, typename _Seq2>
using intseq_join_t = typename intseq_join<_Seq1, _Seq2>::type;


//! create intseq of given length
template<size_t _N>
struct make_intseq
{
  using type = typename intseq_join<typename make_intseq<_N - 1>::type, intseq<_N - 1>>::type;
};

template<>
struct make_intseq<0>
{
  using type = intseq<>;
};

//! extract element from integer sequence
template<std::size_t _Idx, typename _Seq>
struct intseq_element;

template<std::size_t _Idx, template<int ...> class _IntSeq, int Head, int ... Tail>
struct intseq_element<_Idx, _IntSeq<Head, Tail...>>
  : public intseq_element<_Idx - 1, _IntSeq<Tail...>>
{};

template<template<int ...> class _IntSeq, int Head, int ... Tail>
struct intseq_element<0, _IntSeq<Head, Tail ...>>
{
  static constexpr int value = Head;
};


//! sum an integer sequence
template<typename _Seq>
struct intseq_sum;

template<template<int ...> class _IntSeq, int I, int ... Is>
struct intseq_sum<_IntSeq<I, Is...>>
{
  static constexpr int value = I + intseq_sum<_IntSeq<Is...>>::value;
};

template<template<int ...> class _IntSeq>
struct intseq_sum<_IntSeq<>>
{
  static constexpr int value = 0;
};


//! prefix-sum an integer sequence
template<typename _Collected, typename _Remaining, int Sum>
struct intseq_psum;

template<template<int ...> class _IntSeq, int... _Cur, int _Sum>
struct intseq_psum<_IntSeq<_Cur...>, _IntSeq<>, _Sum>
{
  using type = _IntSeq<_Cur...>;
};

template<template<int ...> class _IntSeq, int _First, int _Sum, int... _Cur, int... _Rem>
struct intseq_psum<_IntSeq<_Cur...>, _IntSeq<_First, _Rem...>, _Sum>
  : intseq_psum<_IntSeq<_Cur..., _Sum>, _IntSeq<_Rem...>, _Sum + _First>
{};

template<class _Seq>
using intseq_psum_t = typename intseq_psum<intseq<>, _Seq, 0>::type;

}  // namespace composite
}  // namespace internal

template<size_t _N>
using make_intseq = internal::composite::make_intseq<_N>;

template<size_t _N>
using make_intseq_t = typename internal::composite::make_intseq<_N>::type;

}  // namespace manif

#endif  // _MANIF_MANIF_COMPOSITE_PROPERTIES_H_
