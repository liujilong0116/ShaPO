import torch

file_name = "qwen257_layer_-1_pku_30k_safety"

toxic_vector = torch.load(f"/xxx/ShaPO/classifier_output/{file_name}/toxicity_classifier_head_epoch_10.pt", map_location="cpu")
print(toxic_vector)
vec1 = toxic_vector["mlp.weight"][0]
vec2 = toxic_vector["mlp.weight"][1]

result_vector =  vec2 - vec1

torch.save(result_vector, f"/xxx/ShaPO/classifier_output/{file_name}/toxic_pythia_pku_harmless.pt")

print("保存完成，结果向量大小：", result_vector.shape)
