/* Generated with cbindgen:0.29.0 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define CONJECTURE_SUCCESS 0

#define CONJECTURE_ERROR_NULL_HANDLE -1

#define CONJECTURE_ERROR_INDEX_OUT_OF_BOUNDS -2

#define CONJECTURE_ERROR_INTERNAL -3

#define CONJECTURE_ERROR_INVALID_STRING -4

#define CONJECTURE_ERROR_DATA_OVERFLOW -5

#define CONJECTURE_ERROR_INVALID_PHASE -6

#define CONJECTURE_PHASE_SHRINK 0

typedef enum CPhase {
  Shrink = CONJECTURE_PHASE_SHRINK,
} CPhase;

int32_t conjecture_engine_new(const char *aName,
                              const char *aDatabasePath,
                              uint64_t aSeed,
                              uint64_t aMaxExamples,
                              const enum CPhase *aPhasesPtr,
                              uintptr_t aPhasesLen,
                              void **aResult);

void conjecture_engine_free(void *aHandle);

int32_t conjecture_engine_new_source(void *aHandle,
                                     void **aResult);

int32_t conjecture_engine_count_failing_examples(void *aHandle,
                                                 uintptr_t *aResult);

int32_t conjecture_engine_failing_example(void *aHandle,
                                          uintptr_t aIndex,
                                          void **aResult);

int32_t conjecture_engine_was_unsatisfiable(void *aHandle,
                                            bool *aResult);

int32_t conjecture_engine_finish_overflow(void *aEngineHandle,
                                          void *aDsHandle);

int32_t conjecture_engine_finish_valid(void *aEngineHandle,
                                       void *aDsHandle);

int32_t conjecture_engine_finish_invalid(void *aEngineHandle,
                                         void *aDsHandle);

int32_t conjecture_engine_finish_interesting(void *aEngineHandle,
                                             void *aDsHandle,
                                             uint64_t aLabel);

void conjecture_data_source_free(void *aHandle);

int32_t conjecture_data_source_start_draw(void *aHandle);

int32_t conjecture_data_source_stop_draw(void *aHandle);

int32_t conjecture_data_source_bits(void *aHandle,
                                    uint64_t aNBits,
                                    uint64_t *aResult);

int32_t conjecture_data_source_write(void *aHandle,
                                     uint64_t aValue);

int32_t conjecture_integers_new(void **aResult);

void conjecture_integers_free(void *aHandle);

int32_t conjecture_integers_provide(void *aIntegersHandle,
                                    void *aDsHandle,
                                    int64_t *aResult);

int32_t conjecture_repeat_values_new(uint64_t aMinCount,
                                     uint64_t aMaxCount,
                                     double aExpectedCount,
                                     void **aResult);

void conjecture_repeat_values_free(void *aHandle);

int32_t conjecture_repeat_values_should_continue(void *aRepeatHandle,
                                                 void *aDsHandle,
                                                 bool *aResult);

int32_t conjecture_repeat_values_reject(void *aHandle);

int32_t conjecture_bounded_integers_new(uint64_t aMaxValue,
                                        void **aResult);

void conjecture_bounded_integers_free(void *aHandle);

int32_t conjecture_bounded_integers_provide(void *aBoundedHandle,
                                            void *aDsHandle,
                                            uint64_t *aResult);

uint32_t conjecture_ffi_version(void);
