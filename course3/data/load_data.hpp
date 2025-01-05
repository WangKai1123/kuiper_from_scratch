//
// Created by fss on 22-11-21.
//

#ifndef KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#include <armadillo>
#include <string>
namespace kuiper_infer {

class CSVDataLoader{
    public:
    /**
     * 从csv文件中初始化张量
     * @param file_path csv文件路径
     * @param split_char 分隔符号
     * @return 根据csv文件得到的张量
    */
   static arma::fmat LoadData(const std::string &file_path, char split_char = ',');

    private:
    /**
     * 得到csv文件的尺寸大小
     * @param file_path csv文件路径
     * @param split_char 分隔符号
     * @return csv文件的尺寸大小
     * 
    */
   static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);

};

}

#endif //KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_