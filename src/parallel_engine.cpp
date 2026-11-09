#include <omp.h>
#include <cmath>

extern "C" {

double compute_dot_product(const float* vec_a, const float* vec_b, int dimensions) {
    double sum = 0.0;
    for (int j = 0; j < dimensions; j = j + 1) {
        sum = sum + static_cast<double>(vec_a[j]) * static_cast<double>(vec_b[j]);
    }
    return sum;
}

void search_knn_parallel(
    const float* dataset,
    const int* labels,
    const float* query,
    int num_samples,
    int dimensions,
    int num_classes,
    int num_threads,
    int* out_best_index,
    int* out_best_class,
    float* out_best_score,
    float* out_class_scores,
    int* out_verified_count,
    double* out_elapsed_ms
) {
    int safe_threads = num_threads;
    if (safe_threads < 1) {
        safe_threads = 1;
    }
    omp_set_num_threads(safe_threads);

    double start_time = omp_get_wtime();

    float global_best_score = -1000000.0f;
    int global_best_index = 0;
    int global_best_class = 0;
    int global_verified_count = 0;

    for (int c = 0; c < num_classes; c = c + 1) {
        out_class_scores[c] = -1000000.0f;
    }

    #pragma omp parallel
    {
        float local_best_score = -1000000.0f;
        int local_best_index = 0;
        int local_best_class = 0;
        int local_count = 0;
        float local_class_max[11];

        for (int c = 0; c < 11; c = c + 1) {
            local_class_max[c] = -1000000.0f;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < num_samples; i = i + 1) {
            const float* sample_ptr = dataset + (i * dimensions);
            double dot_val = compute_dot_product(sample_ptr, query, dimensions);
            float score = static_cast<float>(dot_val);
            int label = labels[i];

            local_count = local_count + 1;

            if (label >= 0 && label < 11) {
                if (score > local_class_max[label]) {
                    local_class_max[label] = score;
                }
            }

            if (score > local_best_score) {
                local_best_score = score;
                local_best_index = i;
                local_best_class = label;
            }
        }

        #pragma omp critical
        {
            global_verified_count = global_verified_count + local_count;
            if (local_best_score > global_best_score) {
                global_best_score = local_best_score;
                global_best_index = local_best_index;
                global_best_class = local_best_class;
            }
            for (int c = 0; c < num_classes; c = c + 1) {
                if (c < 11 && local_class_max[c] > out_class_scores[c]) {
                    out_class_scores[c] = local_class_max[c];
                }
            }
        }
    }

    double end_time = omp_get_wtime();

    *out_best_index = global_best_index;
    *out_best_class = global_best_class;
    *out_best_score = global_best_score;
    *out_verified_count = global_verified_count;
    *out_elapsed_ms = (end_time - start_time) * 1000.0;
}

void search_knn_datarace(
    const float* dataset,
    const int* labels,
    const float* query,
    int num_samples,
    int dimensions,
    int num_classes,
    int num_threads,
    int* out_best_index,
    int* out_best_class,
    float* out_best_score,
    float* out_class_scores,
    int* out_corrupted_count,
    double* out_elapsed_ms
) {
    int safe_threads = num_threads;
    if (safe_threads < 2) {
        safe_threads = 4;
    }
    omp_set_num_threads(safe_threads);

    double start_time = omp_get_wtime();

    volatile float shared_best_score = -1000000.0f;
    volatile int shared_best_index = 0;
    volatile int shared_best_class = 0;
    volatile int shared_counter = 0;

    for (int c = 0; c < num_classes; c = c + 1) {
        out_class_scores[c] = 0.0f;
    }

    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < num_samples; i = i + 1) {
        const float* sample_ptr = dataset + (i * dimensions);
        double dot_val = compute_dot_product(sample_ptr, query, dimensions);
        float score = static_cast<float>(dot_val);
        int label = labels[i];

        int temp_counter = shared_counter;
        double dummy_delay = 0.0;
        for (int k = 0; k < 15; k = k + 1) {
            dummy_delay = dummy_delay + 0.0001 * static_cast<double>(k);
        }
        shared_counter = temp_counter + 1;

        if (label >= 0 && label < num_classes) {
            float prev_class_val = out_class_scores[label];
            out_class_scores[label] = (prev_class_val * 0.5f) + (score * 0.5f) + static_cast<float>(dummy_delay * 0.0);
        }

        if (score > shared_best_score) {
            shared_best_score = score;
            shared_best_index = (i + shared_counter) % num_samples;
            shared_best_class = labels[shared_best_index];
        }
    }

    double end_time = omp_get_wtime();

    *out_best_index = shared_best_index;
    *out_best_class = shared_best_class;
    *out_best_score = shared_best_score;
    *out_corrupted_count = shared_counter;
    *out_elapsed_ms = (end_time - start_time) * 1000.0;
}

void run_benchmark_amdahl(
    const float* dataset,
    const int* labels,
    const float* query,
    int num_samples,
    int dimensions,
    int num_classes,
    const int* thread_counts,
    int num_configs,
    int repetitions,
    double* out_times_ms,
    double* out_speedups,
    double* out_efficiencies,
    double* out_karp_flatt
) {
    int best_idx = 0;
    int best_cls = 0;
    float best_scr = 0.0f;
    int ver_cnt = 0;
    double single_elapsed = 0.0;
    float class_scores[11];

    double base_time_t1 = 1.0;

    for (int cfg = 0; cfg < num_configs; cfg = cfg + 1) {
        int p_threads = thread_counts[cfg];
        double total_time = 0.0;

        for (int rep = 0; rep < repetitions; rep = rep + 1) {
            search_knn_parallel(
                dataset,
                labels,
                query,
                num_samples,
                dimensions,
                num_classes,
                p_threads,
                &best_idx,
                &best_cls,
                &best_scr,
                class_scores,
                &ver_cnt,
                &single_elapsed
            );
            total_time = total_time + single_elapsed;
        }

        double avg_time = total_time / static_cast<double>(repetitions);
        out_times_ms[cfg] = avg_time;

        if (cfg == 0 || p_threads == 1) {
            base_time_t1 = avg_time;
        }

        double speedup = base_time_t1 / avg_time;
        double efficiency = speedup / static_cast<double>(p_threads);

        out_speedups[cfg] = speedup;
        out_efficiencies[cfg] = efficiency;

        double kf_val = 0.0;
        if (p_threads > 1 && speedup > 0.0) {
            double inv_s = 1.0 / speedup;
            double inv_p = 1.0 / static_cast<double>(p_threads);
            kf_val = (inv_s - inv_p) / (1.0 - inv_p);
            if (kf_val < 0.0) {
                kf_val = 0.0;
            }
        }
        out_karp_flatt[cfg] = kf_val;
    }
}

}
