#include "data/tensor.hpp"
#include "glog/logging.h"
#include <memory>
#include <numeric>

namespace kuiper_infer{
    Tensor<float>::Tensor(uint32_t channels,uint32_t rows,uint32_t cols){
        data_ = arma::fcube(rows,cols,channels);
        if(channels == 1 && rows ==1){
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }else if(channels ==1){
            this->raw_shapes_ = std::vector<uint32_t>{rows,cols};
        }else{
            this->raw_shapes_ = std::vector<uint32_t>{channels,rows,cols};
        }
    }

    Tensor<float>::Tensor(uint32_t size){
        data_ = arma::fcube(1,size,1);
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }
    Tensor<float>::Tensor(uint32_t rows,uint32_t cols){
        data_ = arma::fcube(rows,cols,1);
        this->raw_shapes_ = std::vector<uint32_t>{rows,cols};
    }
    Tensor<float>::Tensor(const std::vector<uint32_t>& shapes){
        CHECK(!shapes.empty() && shapes.size() <= 3);

        uint32_t remaining = 3 - shapes.size();
        //初始化一个有三个元素的一维向量，每个元素都是1
        std::vector<uint32_t> shapes_(3,1);
        //shape中的元素拷贝到shapes_中
        std::copy(shapes.begin(),shapes.end(),shapes_.begin()+remaining);
        uint32_t channels = shapes_.at(0);
        uint32_t rows = shapes_.at(1);
        uint32_t cols = shapes_.at(2);

        data_ = arma::fcube(rows,cols,channels);
        if(channels == 1 && rows ==1){
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }else if(channels ==1){
            this->raw_shapes_ = std::vector<uint32_t>{rows,cols};
        }else{
            this->raw_shapes_ = std::vector<uint32_t>{channels,rows,cols};
        }
    }

    Tensor<float>::Tensor(const Tensor& tensor){
        if (this != &tensor) {
            this->data_ = tensor.data_;
            this->raw_shapes_ = tensor.raw_shapes_;
        }
    }

    Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
         if (this != &tensor) {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
        }     
    }

    Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
        if(this != &tensor){
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
        }
        return *this;
    }

    Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) {
        if(this != &tensor){
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
        }
        return *this;
    }

    uint32_t Tensor<float>::rows() const {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    uint32_t Tensor<float>::cols() const {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    uint32_t Tensor<float>::channels() const {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    uint32_t Tensor<float>::size() const {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    void Tensor<float>::set_data(const arma::fcube& data) {
    CHECK(data.n_rows == this->data_.n_rows)
        << data.n_rows << " != " << this->data_.n_rows;
    CHECK(data.n_cols == this->data_.n_cols)
        << data.n_cols << " != " << this->data_.n_cols;
    CHECK(data.n_slices == this->data_.n_slices)
        << data.n_slices << " != " << this->data_.n_slices;
    this->data_ = data;
    }

    bool Tensor<float>::empty() const { return this->data_.empty(); }

    float Tensor<float>::index(uint32_t offset) const {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    float& Tensor<float>::index(uint32_t offset) {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }


    std::vector<uint32_t> Tensor<float>::shapes() const {
        CHECK(!this->data_.empty());
        return {this->channels(), this->rows(), this->cols()};
    }

    arma::fcube& Tensor<float>::data() { return this->data_; }

    const arma::fcube& Tensor<float>::data() const { return this->data_; }

    arma::fmat& Tensor<float>::slice(uint32_t channel) {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    void Tensor<float>::Padding(const std::vector<uint32_t>& pads,float padding_value) {
        // TODO: Implement padding
        CHECK(!this->data_.empty());
        CHECK_EQ(pads.size(),4);
        //四周的维度
        uint32_t  pad_rows1 = pads.at(0); // up
        uint32_t  pad_rows2 = pads.at(1); // down
        uint32_t  pad_cols1 = pads.at(2); // left
        uint32_t  pad_cols2 = pads.at(3);   // right

        //填充代码
        uint32_t pad_rows = this->rows() + pad_rows1 + pad_rows2;
        uint32_t pad_cols = this->cols() + pad_cols1 + pad_cols2;
        uint32_t channels = this->channels();
        // 创建一个新的tensor表示填充之后的数据
        Tensor<float> padded(pad_channels, pad_rows, pad_cols);


    }


}
