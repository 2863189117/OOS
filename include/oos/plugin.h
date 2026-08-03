#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define OOS_PLUGIN_EXPORT __declspec(dllexport)
#else
#define OOS_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OOS_SURFACE_PLUGIN_ABI_V3 3u
#define OOS_FUNCTION_OPERATOR_ABI_V1 1u

typedef enum oos_surface_locality_v3 {
  OOS_SURFACE_LOCAL_V3 = 1,
  OOS_SURFACE_NONLOCAL_V3 = 2
} oos_surface_locality_v3;

typedef struct oos_string_view_v1 {
  const char* data;
  size_t size;
} oos_string_view_v1;

typedef struct oos_plugin_validation_v1 {
  int32_t status;
  oos_string_view_v1 message;
} oos_plugin_validation_v1;

typedef struct oos_operator_sink_v1 {
  void* user_data;
  int32_t (*write_dataset_f64)(void*, oos_string_view_v1, const double*,
                               const uint64_t*, size_t);
  int32_t (*write_dataset_u64)(void*, oos_string_view_v1, const uint64_t*,
                               const uint64_t*, size_t);
  int32_t (*write_metadata_json)(void*, oos_string_view_v1);
} oos_operator_sink_v1;

typedef struct oos_surface_hit_v3 {
  uint64_t surface_element;
  uint32_t incident_side;
  double barycentric[3];
  double point_local_mm[3];
  double direction_local[3];
  double reference_axis_local[3];
  double stokes[4];
  double incident_refractive_index;
  double transmitted_refractive_index;
} oos_surface_hit_v3;

typedef struct oos_local_interaction_v3 {
  double specular_reflection;
  double specular_transmission;
  double lambertian_reflection;
  double detection;
  double absorption;
} oos_local_interaction_v3;

typedef struct oos_surface_plugin_v3 {
  uint32_t abi_version;
  uint32_t locality;
  oos_string_view_v1 name;
  oos_string_view_v1 version;
  oos_plugin_validation_v1 (*validate)(oos_string_view_v1 config_json,
                                       double energy_eV);
  int32_t (*build_operator)(oos_string_view_v1 config_json, double energy_eV,
                            const oos_operator_sink_v1* sink);
  int32_t (*interact_local)(oos_string_view_v1 config_json, double energy_eV,
                            const oos_surface_hit_v3* hit,
                            oos_local_interaction_v3* output);
  int32_t (*deposit_nonlocal)(oos_string_view_v1 config_json, double energy_eV,
                              const oos_surface_hit_v3* hit,
                              uint64_t* state_indices, double* weights,
                              size_t capacity, size_t* count);
  size_t maximum_deposition_entries;
} oos_surface_plugin_v3;

typedef const oos_surface_plugin_v3* (*oos_get_surface_plugin_v3_fn)(void);

/*
 * Matrix-free linear operator implemented by a custom nonlocal surface.
 *
 * The surface ABI above owns incident deposition and the intrinsic egress
 * basis.  This ABI maps batches of weights in that input basis to retained
 * intrinsic state, surface-relative egress coefficients, and intrinsic
 * losses.  The core maps egress coefficients through the external geometry.
 *
 * All arrays are contiguous row-major float64.  Implementations must be
 * deterministic, linear, non-negative, and passive.  The core validates
 * those properties with probe batches before accepting an operator cache.
 */
typedef struct oos_function_operator_descriptor_v1 {
  uint32_t abi_version;
  uint64_t input_state_count;
  uint64_t retained_state_count;
  uint64_t egress_count;
  uint64_t loss_count;
  double contraction_bound;
  uint32_t supports_cpu;
  uint32_t supports_cuda;
} oos_function_operator_descriptor_v1;

typedef struct oos_function_operator_audit_v1 {
  double input_weight;
  double retained_weight;
  double egress_weight;
  double loss_weight;
  double closure_error;
} oos_function_operator_audit_v1;

typedef struct oos_function_operator_v1 {
  uint32_t abi_version;
  oos_string_view_v1 name;
  oos_string_view_v1 version;
  oos_plugin_validation_v1 (*validate)(oos_string_view_v1 config_json,
                                       double energy_eV);
  int32_t (*create)(oos_string_view_v1 config_json, double energy_eV,
                    void** instance,
                    oos_function_operator_descriptor_v1* descriptor);
  void (*destroy)(void* instance);
  int32_t (*apply_cpu)(void* instance, uint64_t batch,
                       const double* input, double* retained,
                       double* egress, double* losses,
                       oos_function_operator_audit_v1* audit);
  int32_t (*prepare_cuda)(void* instance, int32_t device);
  int32_t (*apply_cuda)(void* instance, uint64_t batch,
                        const double* device_input,
                        double* device_retained, double* device_egress,
                        double* device_losses, void* cuda_stream,
                        /* May be null for asynchronous production solves. */
                        oos_function_operator_audit_v1* host_audit);
} oos_function_operator_v1;

typedef const oos_function_operator_v1*
    (*oos_get_function_operator_v1_fn)(void);

#ifdef __cplusplus
}
#endif
