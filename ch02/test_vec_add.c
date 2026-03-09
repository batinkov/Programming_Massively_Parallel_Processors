#include <assert.h>
#include <string.h>
#include "vec_add.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(fn) do { \
    tests_run++; \
    printf("  %-50s", #fn); \
    fn(); \
    tests_passed++; \
    printf("PASS\n"); \
} while (0)

static void test_vec_add_basic(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float c[3];

    vec_add_cpu(a, b, c, 3);

    assert(c[0] == 5.0f);
    assert(c[1] == 7.0f);
    assert(c[2] == 9.0f);
}

static void test_vec_add_zeros(void) {
    float a[] = {0.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 0.0f, 0.0f};
    float c[3];

    vec_add_cpu(a, b, c, 3);

    assert(c[0] == 0.0f);
    assert(c[1] == 0.0f);
    assert(c[2] == 0.0f);
}

static void test_vec_add_negatives(void) {
    float a[] = {-1.0f, -2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, -3.0f};
    float c[3];

    vec_add_cpu(a, b, c, 3);

    assert(c[0] == 0.0f);
    assert(c[1] == 0.0f);
    assert(c[2] == 0.0f);
}

static void test_vec_add_single_element(void) {
    float a[] = {42.0f};
    float b[] = {58.0f};
    float c[1];

    vec_add_cpu(a, b, c, 1);

    assert(c[0] == 100.0f);
}

static void test_vec_add_large(void) {
    int n = 10000;
    float *a = malloc((size_t)n * sizeof(float));
    float *b = malloc((size_t)n * sizeof(float));
    float *c = malloc((size_t)n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }

    vec_add_cpu(a, b, c, n);

    for (int i = 0; i < n; i++) {
        assert(c[i] == (float)n);
    }

    free(a);
    free(b);
    free(c);
}

static void test_generate_array_deterministic(void) {
    int n = 1000;
    float *a1 = malloc((size_t)n * sizeof(float));
    float *a2 = malloc((size_t)n * sizeof(float));

    generate_array(a1, n, 42);
    generate_array(a2, n, 42);

    for (int i = 0; i < n; i++) {
        assert(a1[i] == a2[i]);
    }

    free(a1);
    free(a2);
}

static void test_generate_array_different_seeds(void) {
    int n = 1000;
    float *a1 = malloc((size_t)n * sizeof(float));
    float *a2 = malloc((size_t)n * sizeof(float));

    generate_array(a1, n, 42);
    generate_array(a2, n, 137);

    int identical = 1;
    for (int i = 0; i < n; i++) {
        if (a1[i] != a2[i]) {
            identical = 0;
            break;
        }
    }
    assert(!identical);

    free(a1);
    free(a2);
}

static void test_generate_array_range(void) {
    int n = 10000;
    float *a = malloc((size_t)n * sizeof(float));

    generate_array(a, n, 99);

    for (int i = 0; i < n; i++) {
        assert(a[i] >= 0.0f && a[i] <= 1.0f);
    }

    free(a);
}

static void test_verify_result_pass(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};

    assert(verify_result(a, b, 3) == 1);
}

static void test_verify_result_fail(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 4.0f};

    assert(verify_result(a, b, 3) == 0);
}

static void test_verify_result_exact_match(void) {
    float a[] = {1.0f / 3.0f, 2.0f / 7.0f, 3.0f / 11.0f};
    float b[] = {1.0f / 3.0f, 2.0f / 7.0f, 3.0f / 11.0f};

    assert(verify_result(a, b, 3) == 1);
}

static void test_parse_args_defaults(void) {
    int n, num_runs;
    char *argv[] = {"test"};
    parse_args(1, argv, &n, &num_runs);

    assert(n == DEFAULT_N);
    assert(num_runs == DEFAULT_RUNS);
}

static void test_parse_args_custom(void) {
    int n, num_runs;
    char *argv[] = {"test", "-n", "5000", "-r", "10"};
    parse_args(5, argv, &n, &num_runs);

    assert(n == 5000);
    assert(num_runs == 10);
}

int main(void) {
    printf("Running vec_add tests:\n");

    RUN_TEST(test_vec_add_basic);
    RUN_TEST(test_vec_add_zeros);
    RUN_TEST(test_vec_add_negatives);
    RUN_TEST(test_vec_add_single_element);
    RUN_TEST(test_vec_add_large);
    RUN_TEST(test_generate_array_deterministic);
    RUN_TEST(test_generate_array_different_seeds);
    RUN_TEST(test_generate_array_range);
    RUN_TEST(test_verify_result_pass);
    printf("  (next test expects a 'Mismatch' message on stderr)\n");
    RUN_TEST(test_verify_result_fail);
    RUN_TEST(test_verify_result_exact_match);
    RUN_TEST(test_parse_args_defaults);
    RUN_TEST(test_parse_args_custom);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
