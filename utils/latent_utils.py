import warnings
import torch
import torch.nn.functional as F


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def get_delta_weight(self, adapter) -> torch.Tensor:
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # Handle CPU + float16/bfloat16 case
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight
    latent_weight = self.default_lora_latent_mapping.weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()
        latent_weight = latent_weight.float()
    else:
        # Ensure all tensors are same dtype (bfloat16 in your case)
        weight_A = weight_A.to(dtype)
        weight_B = weight_B.to(dtype)
        latent_weight = latent_weight.to(dtype)

    output_tensor = transpose(
        weight_B @ latent_weight @ weight_A,
        self.fan_in_fan_out
    ) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)
        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)
        self.default_lora_latent_mapping.weight.data = latent_weight.to(dtype)

    return output_tensor


def forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        result += (
            self.lora_B[self.active_adapter[0]](
                self.default_lora_latent_mapping(
                    self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
                )
            )
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result

