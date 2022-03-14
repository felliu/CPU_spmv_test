#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <algorithm>
#include <tuple>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <type_traits>

/**
 * Reads an array from a binary file. The file is stored as |sz|data|, where sz is an unsigned 64-bit int.
 * If two template parameters are specified, the values in the output vector will be cast to type of the
 * second template argument.
 **/
template <typename ValueType, typename OutType = ValueType>
std::vector<OutType> read_binary_file(const std::string& path)
{
    std::ifstream in_file(path, std::ios::in | std::ios::binary);
    size_t sz;
    in_file.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    std::vector<ValueType> data_in(sz);
    in_file.read(reinterpret_cast<char*>(data_in.data()), sizeof(ValueType) * sz);
    std::vector<OutType> out_vec;
    out_vec.reserve(sz);

    //Cast to the right output form. Not strictly necessary if U == T.
    std::transform(data_in.cbegin(), data_in.cend(), std::back_inserter(out_vec),
                   [](ValueType val){ return static_cast<OutType>(val); });

    return out_vec;
}

template <typename T>
void dump_vector_to_file(const std::vector<T>& vec, const std::string& path, bool append=false)
{
    std::ofstream outfile;
    if (append)
        outfile.open(path, std::ios::binary | std::ios::app);
    else
        outfile.open(path, std::ios::binary | std::ios::out);

    size_t sz = vec.size();
    outfile.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    outfile.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * sz);
}


template <typename T>
std::tuple<std::vector<std::vector<T>>, uint64_t, uint64_t>
read_crlsp(const std::string& path)
{
    std::ifstream istream(path, std::ios::in | std::ios::binary);
    std::vector<std::vector<T>> data;
    uint64_t sz = 0;
    uint64_t rows, cols;
    istream.read(reinterpret_cast<char*>(&rows), sizeof(uint64_t));
    istream.read(reinterpret_cast<char*>(&cols), sizeof(uint64_t));
    for (int i = 0; i < cols; ++i) {
        istream.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        auto& new_el = data.emplace_back(sz);
        istream.read(reinterpret_cast<char*>(new_el.data()), sizeof(T) * sz);
    }

    return std::make_tuple(data, rows, cols);
}

template <typename OutType, typename InputType = OutType>
std::vector<std::vector<OutType>> read_data(const std::string& path)
{
    std::ifstream istream(path, std::ios::in | std::ios::binary);
    std::vector<std::vector<OutType>> data;
    uint64_t sz = 0;
    while (istream.read(reinterpret_cast<char*>(&sz), sizeof(sz))) {
        auto& new_vec = data.emplace_back();
        new_vec.reserve(sz);

        std::vector<InputType> tmp(sz);
        istream.read(reinterpret_cast<char*>(tmp.data()), sizeof(InputType) * sz);
        //Convert data to the output type.
        std::transform(tmp.cbegin(), tmp.cend(), std::back_inserter(new_vec),
            [](InputType val) { return static_cast<OutType>(val); });
    }
    return data;
}

template<typename ValueType, typename IndexType>
struct CSR_matrix {
    std::vector<ValueType> data;
    std::vector<IndexType> col_ind;
    std::vector<IndexType> row_ind;
    size_t rows, cols;

    void dump_to_binary(const std::string& path) {
        std::ofstream outfile(path, std::ios::binary | std::ios::out);
        outfile.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        outfile.write(reinterpret_cast<char*>(&cols), sizeof(cols));
        outfile.close();
        dump_vector_to_file(data, path, true);
        dump_vector_to_file(col_ind, path, true);
        dump_vector_to_file(row_ind, path, true);
    }

    static CSR_matrix<ValueType, IndexType> read_from_binary_file(const std::string& path) {
        size_t sz;
        CSR_matrix<ValueType, IndexType> mat;
        std::ifstream infile(path, std::ios::binary | std::ios::in);

        infile.read(reinterpret_cast<char*>(&mat.rows), sizeof(size_t));
        infile.read(reinterpret_cast<char*>(&mat.cols), sizeof(size_t));
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.data.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.data.data()), sizeof(ValueType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.col_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.col_ind.data()), sizeof(IndexType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.row_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.row_ind.data()), sizeof(IndexType) * sz);
        return mat;
    }

    static CSR_matrix<ValueType, IndexType> read_from_binary_file_old(const std::string& path) {
        size_t sz;
        CSR_matrix<ValueType, IndexType> mat;
        std::ifstream infile(path, std::ios::binary | std::ios::in);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.data.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.data.data()), sizeof(ValueType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.col_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.col_ind.data()), sizeof(IndexType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.row_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.row_ind.data()), sizeof(IndexType) * sz);
        return mat;
    }
};


template<typename ValueType, typename IndexType>
struct CSC_matrix {
    std::vector<ValueType> data;
    std::vector<IndexType> row_ind;
    std::vector<IndexType> col_ind;
    size_t rows, cols;

    void dump_to_binary(const std::string& path) {
        std::ofstream outfile(path, std::ios::binary | std::ios::out);
        outfile.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        outfile.write(reinterpret_cast<char*>(&cols), sizeof(cols));
        outfile.close();
        dump_vector_to_file(data, path, true);
        dump_vector_to_file(row_ind, path, true);
        dump_vector_to_file(col_ind, path, true);
    }

    static CSC_matrix<ValueType, IndexType> read_from_binary_file(const std::string& path) {
        size_t sz;
        CSC_matrix<ValueType, IndexType> mat;
        std::ifstream infile(path, std::ios::binary | std::ios::in);

        infile.read(reinterpret_cast<char*>(&mat.rows), sizeof(size_t));
        infile.read(reinterpret_cast<char*>(&mat.cols), sizeof(size_t));
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.data.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.data.data()), sizeof(ValueType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.col_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.row_ind.data()), sizeof(IndexType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.row_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.col_ind.data()), sizeof(IndexType) * sz);
        return mat;
    }

    /*
    static CSC_matrix read_from_binary_file_old(const std::string& path) {
        size_t sz;
        CSC_matrix mat;
        std::ifstream infile(path, std::ios::binary | std::ios::in);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.data.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.data.data()), sizeof(ValueType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.row_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.row_ind.data()), sizeof(IndexType) * sz);
        infile.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        mat.col_ind.resize(sz);
        infile.read(reinterpret_cast<char*>(mat.col_ind.data()), sizeof(IndexType) * sz);
        return mat;
    }
    */
};

template <typename ValueType, typename IndexType>
CSC_matrix<ValueType, IndexType> convert_to_CSC(const std::vector<std::vector<ValueType>>& data,
                                const std::vector<std::vector<IndexType>>& CRLSP_arr,
                                uint32_t rows) {
    uint64_t total_size = std::accumulate(data.begin(), data.end(), 0ULL,
        [](auto acc, const auto& vec){ return acc + vec.size(); });
    CSC_matrix<ValueType, IndexType> CSC;
    CSC.data.reserve(total_size);
    CSC.col_ind.reserve(data.size());

    auto col_start_idx = 0;
    //Populate the col_ind array and data
    for (auto& col : data) {
        std::copy(col.begin(), col.end(), std::back_inserter(CSC.data));
        CSC.col_ind.push_back(col_start_idx);
        col_start_idx += col.size();
    }

    CSC.col_ind.push_back(col_start_idx);

    //Populate the row indices.
    //The CRLSP array stores the starting index of the row at even indexes, followed by a cumulative run length value.
    //Subtracting the current CRL value with the previous one gives the length of the current run length
    //(number of consecutive elements present in the matrix)
    CSC.row_ind.resize(total_size);
    auto iter = CSC.row_ind.begin();
    for (const auto& crlsp : CRLSP_arr) {
        for (int i = 0; i < crlsp.size(); i += 2) {
            auto run_length = static_cast<IndexType>(0);
            auto starting_row = crlsp[i];
            if (i == 0)
                run_length = crlsp[i + 1];
            else
                run_length = crlsp[i + 1] - crlsp[i - 1];

            std::iota(iter, iter + run_length, starting_row);
            iter += run_length;
        }
    }

    return CSC;
}

#endif
