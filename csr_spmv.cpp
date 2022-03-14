#define USE_BLAZE

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <tuple>

#ifndef USE_BLAZE
#include <mkl.h>
#endif
#include <Eigen/Sparse>
#include <Eigen/Core>
#ifdef USE_BLAZE
#include <blaze/Blaze.h>
#endif

#include "conversions.h"

template <typename T>
std::vector<Eigen::Triplet<T, int>>
csr_to_triplet(int rows, int cols,
               const T* vals, const int* col_idxs, const int* row_ptrs) {
    const int nnz = row_ptrs[rows];
    std::vector<Eigen::Triplet<T>> triplets;
    triplets.reserve(nnz);

    for (int row = 0; row < rows; ++row)
        for (int idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            triplets.push_back(Eigen::Triplet<T, int>(row, col_idxs[idx], vals[idx]));
        }

    return triplets;
}

#ifdef USE_BLAZE
template <typename T>
blaze::CompressedMatrix<T>
init_blaze_mat(int rows, int cols, const T* vals, const int* col_idxs, const int* row_ptrs) {
    blaze::CompressedMatrix<T> mat(rows, cols);
    int nnz = row_ptrs[rows];
    mat.reserve(nnz);
    for (int row = 0; row < rows; ++row) {
        for (int idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            mat.append(row, col_idxs[idx], vals[idx]);
        }
        mat.finalize(row);
    }
    return mat;
}
#endif

template <typename T>
std::tuple<double, double>
compute_avg_stddev(const std::vector<T>& vals) {
    const auto num_vals = vals.size();
    const double total = std::accumulate(vals.cbegin(), vals.cend(), 0.0,
                                         [](double acc, int64_t val){return acc + static_cast<double>(val);});
    const double avg = total / static_cast<double>(num_vals);
    const double sq_sum = std::accumulate(vals.cbegin(), vals.cend(), 0.0,
                                          [avg](double acc, int64_t val){
                                            return acc + static_cast<double>(val - avg) * static_cast<double>(val - avg);
                                          });
    const double stddev = std::sqrt(sq_sum / static_cast<double>(num_vals));
    return std::make_tuple(avg, stddev);
}

#ifndef USE_BLAZE
bool check_err(sparse_status_t err)
{
    switch(err)
    {
        case SPARSE_STATUS_SUCCESS:
            return true;
        case SPARSE_STATUS_NOT_INITIALIZED:
            std::cerr << "Non initialized array encountered.\n";
            return false;
        case SPARSE_STATUS_ALLOC_FAILED:
            std::cerr << "Memory allocation failed.\n";
            return false;
        case SPARSE_STATUS_INVALID_VALUE:
            std::cerr << "Invalid parameter value.\n";
            return false;
        case SPARSE_STATUS_EXECUTION_FAILED:
            std::cerr << "Execution failed.\n";
            return false;
        case SPARSE_STATUS_INTERNAL_ERROR:
            std::cerr << "Internal error in Sparse BLAS.\n";
            return false;
        case SPARSE_STATUS_NOT_SUPPORTED:
            std::cerr << "Operation not supported.\n";
            return false;
    }
}

void CSR_SpMV_MKL(CSR_matrix<float, int32_t>& mat, const std::vector<float>& x, std::vector<float>& y, int num_runs)
{
    using namespace std::chrono;
    //mkl_verbose(1);
    //MKL:s representation of CSR splits the row_ptr array into two arrays. One for where each row begins, and one for where it ends.
    std::vector<int32_t> row_begin;
    //Where each row begins is just the entire row_ind array, without the last element.
    std::copy(mat.row_ind.begin(), mat.row_ind.end() - 1, std::back_inserter(row_begin));

    std::vector<int32_t> row_end;
    //Where each row ends is just the entire row_ind array, without the first element.
    std::copy(mat.row_ind.begin() + 1, mat.row_ind.end(), std::back_inserter(row_end));

    sparse_status_t err;
    sparse_matrix_t A;
    err = mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO,
                                  mat.rows, mat.cols,
                                  mat.row_ind.data(), mat.row_ind.data() + 1,
                                  mat.col_ind.data(), mat.data.data());
    assert(check_err(err));

    std::cerr << "Sparse matrix created...\n";

    std::vector<int64_t> durations;
    matrix_descr desc = {.type=SPARSE_MATRIX_TYPE_GENERAL};
    for (int i = 0; i < num_runs; ++i) {
        auto start = high_resolution_clock::now();
        err = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, desc, x.data(), 1.0, y.data());
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        durations.push_back(duration);
        assert(check_err(err));
    }

    auto [avg, stddev] = compute_avg_stddev(durations);

    std::cout << "Average elapsed time (MKL): " << avg<< " ms" << std::endl;
    std::cout << "Standard deviation (MKL): " << stddev << " ms" << std::endl;
}
#endif

void CSR_SpMV_Eigen(const CSR_matrix<float, int32_t>& mat,
                    const std::vector<float>& x,
                    std::vector<float>& y,
                    int num_runs) {
    using namespace std::chrono;
    Eigen::setNbThreads(12);
    Eigen::SparseMatrix<float, Eigen::RowMajor> eigen_mat(mat.rows, mat.cols);
    {
        std::vector<Eigen::Triplet<float, int>> triplets =
            csr_to_triplet(mat.rows, mat.cols, mat.data.data(), mat.col_ind.data(), mat.row_ind.data());
        eigen_mat.setFromTriplets(triplets.cbegin(), triplets.cend());
    }
    Eigen::Map<const Eigen::VectorXf> x_map(x.data(), x.size());
    Eigen::Map<Eigen::VectorXf> y_map(y.data(), y.size());
    std::vector<int64_t> durations;
    for (int i = 0; i < num_runs; ++i) {
        auto start = high_resolution_clock::now();
        y_map = eigen_mat * x_map;
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        durations.push_back(duration);
    }

    auto [avg, stddev] = compute_avg_stddev(durations);
    std::cout << "Average elasped time (Eigen):" << avg<< " ms" << std::endl;
    std::cout << "Standard deviation (Eigen): " << stddev << " ms" << std::endl;
}

#ifdef USE_BLAZE
void CSR_SpMV_blaze(const CSR_matrix<float, int32_t>& mat,
                    const std::vector<float>& x,
                    std::vector<float>& y,
                    int num_runs) {
    using namespace std::chrono;
    blaze::CompressedMatrix<float> mat_bz =
        init_blaze_mat(mat.rows, mat.cols,
                       &mat.data[0], &mat.col_ind[0],
                       &mat.row_ind[0]);
    blaze::CustomVector<const float, blaze::unaligned, blaze::unpadded, blaze::columnVector>
    x_bz(&x[0], x.size());
    blaze::CustomVector<float, blaze::unaligned, blaze::unpadded, blaze::columnVector>
    y_bz(&y[0], y.size());

    std::vector<int64_t> durations;
    for (int i = 0; i < num_runs; ++i) {
        auto start = high_resolution_clock::now();
        y_bz = mat_bz * x_bz;
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        durations.push_back(duration);
    }
    auto [avg, stddev] = compute_avg_stddev(durations);
    std::cout << "Average elasped time (Blaze):" << avg << " ms" << std::endl;
    std::cout << "Standard deviation (Blaze): " << stddev << " ms" << std::endl;
}
#endif

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "Usage: ./" << argv[0] << " [file_name]\n";
        return -1;
    }

    std::string file_name(argv[1]);
    CSR_matrix<float, int32_t> mat;
    {
        std::cerr << "Reading binary matrix...\n";
        auto mat_tmp = CSR_matrix<uint16_t, int32_t>::read_from_binary_file(file_name);
        std::cerr << "Converting to float...\n";
        mat.data.reserve(mat_tmp.data.size());
        std::transform(mat_tmp.data.begin(), mat_tmp.data.end(), std::back_inserter(mat.data),
                       [](uint16_t short_val){return static_cast<float>(short_val);});

        /*mat.col_ind.reserve(mat_tmp.col_ind.size());
        std::copy(mat_tmp.col_ind.begin(), mat_tmp.col_ind.end(), std::back_inserter(mat.col_ind));*/
        mat.col_ind = std::move(mat_tmp.col_ind);

        /*mat.row_ind.reserve(mat_tmp.row_ind.size());
        std::copy(mat_tmp.row_ind.begin(), mat_tmp.row_ind.end(), std::back_inserter(mat.row_ind));*/
        mat.row_ind = std::move(mat_tmp.row_ind);

        mat.rows = mat_tmp.rows;
        mat.cols = mat_tmp.cols;
    }

    std::vector<float> x(mat.cols);
    std::vector<float> y(mat.rows);

    std::cerr << "Entering SpMV...\n";
    //CSR_SpMV_MKL(mat, x, y, 100);

    //std::vector<float> y_eigen(mat.rows);
    //CSR_SpMV_Eigen(mat, x, y_eigen, 1);
    //std::vector<float> y_bz(mat.rows);
    CSR_SpMV_blaze(mat, x, y, 10);

    /*double sq_diff_total = 0.0;
    for (int i = 0; i < mat.rows; ++i) {
        sq_diff_total += (y_bz[i] - y[i]) * (y_bz[i] - y[i]);
    }

    std::cerr << "Sq diff: " << sq_diff_total << "\n";*/
    return 0;
}
