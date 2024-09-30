#include <cmath>
#include <string>
#include <vector>
// third party 
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <openvino/openvino.hpp>

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};

class Detector {
public:
  Detector(const std::string &label_path,
           const std::string &model_path, 
           const std::string &device_name,
           const std::vector<std::string> &ignore_classes,
           float conf_threshold = 0.8,
           float nms_thre = 0.4,
           int top_k = 50);


  std::vector<Armor> detect(const cv::Mat &input) noexcept;
private:
  void init();

  cv::Mat preprocessImage(const cv::Mat &input) noexcept;

  cv::Mat infer(const cv::Mat &input) noexcept;

  std::vector<Armor> postprocessOutput(const cv::Mat &input) noexcept;

  // void nmsMergeSortedBboxes() noexcept;

  std::string model_path_;
  std::string device_name_;
  float conf_threshold_;
  float nms_threshold_;
  int top_k_;
  std::vector<int> strides_;
  std::vector<GridAndStride> grid_strides_;

  Eigen::Matrix3f transform_matrix_;
  std::mutex mtx_;
  std::unique_ptr<ov::Core> ov_core_;
  std::unique_ptr<ov::CompiledModel> compiled_model_;

  std::vector<std::string> class_names_;
  std::vector<std::string> ignore_classes_;
};
