// =============================================================================
// Instrumentation hooks for the private Lab (count / profile modes).
//
// In the product these expand to nothing: the timing build is byte-identical
// to a build without this header (checked by disassembly hash in the Lab).
// The Lab compiles the same sources with -DMAXMULSK_LAB_HOOKS and supplies
// "maxmulsk_lab_hooks_impl.hpp" on its include path; that header decides,
// per scope and at compile time, whether a hook counts, times, or vanishes.
//
// Scope names (contract §5.1): allocation, pack_a, pack_a_read_transform,
// pack_a_transform_store, pack_b, za_init, compute, writeback. The macros are
// placed by hand at the boundaries the kernel already has; nothing is moved
// into helpers or fenced to make a timer fit.
// =============================================================================
#pragma once
#if defined(MAXMULSK_LAB_HOOKS)
#include "maxmulsk_lab_hooks_impl.hpp"
#else
#define MAXMULSK_HOOK_EVENT(scope) ((void)0)
#define MAXMULSK_HOOK_BEGIN(scope) ((void)0)
#define MAXMULSK_HOOK_END(scope)   ((void)0)
#endif
