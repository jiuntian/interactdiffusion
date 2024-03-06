import torch


def add_additional_channels(state_dict, num_additional_channels):
    "state_dict should be just from unet model, not the entire SD or GLIGEN"

    if num_additional_channels != 0:
        new_conv_weight = torch.zeros(320, 4+num_additional_channels, 3, 3 )

        for key,value in state_dict.items():
            if key == "input_blocks.0.0.weight":
                old_conv_weight = value
                new_conv_weight[:,0:4,:,:] = old_conv_weight
                state_dict[key] = new_conv_weight
