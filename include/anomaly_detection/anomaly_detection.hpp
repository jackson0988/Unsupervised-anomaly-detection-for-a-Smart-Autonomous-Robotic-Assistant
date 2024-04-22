
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <torch/script.h>
#include <vector>

class AnomalyDetector
{
public:
    AnomalyDetector(const std::string module_path) : module_path(module_path)
    {
        try {
            module = torch::jit::load(module_path.c_str());
        }
        catch (const c10::Error& e) {
            throw new std::exception(e);
        }
        module.to(at::kCUDA);
        img_data.push_back(at::zeros({ 1, 3, 128, 128 }, at::ScalarType::Float).to(at::kCUDA));
    };

    float compute(std::vector<unsigned char>& inputs)
    {
        
        auto img = at::from_blob(inputs.data(), { 128, 128, 3 }, at::TensorOptions(at::ScalarType::Byte)).to(at::kCUDA, false, true);;
        /*
            img = (img / 255 - 0.5) * 2
            img = img.permute(2, 0, 1)                         
            img = torch.unsqueeze(img, 0)
        */
        img = 2.0f * (-0.5f + img / 255.0f);
        img                          = img.permute({ 2, 0, 1 });
        img                          = at::unsqueeze(img, 0);
        // store data in input vector
        img_data.front()             = img;
        at::Tensor reconstructed_img = module.forward(img_data).toTensor();
        /*
            reconstructed_img = (reconstructed_img / 2 + 0.5 * 255)
            img = (img / 2 + 0.5) * 255
            ano_score = torch.mean(torch.abs(reconstructed_img-img),dim = [1,2,3])
        */
        img       = 255.0f * (0.5f + img / 2.0f);
        return at::mean(at::abs(reconstructed_img - img), { 1, 2, 3 }).item<float>();
    };

private:
    std::string                     module_path;
    torch::jit::script::Module      module;
    std::vector<torch::jit::IValue> img_data;
};