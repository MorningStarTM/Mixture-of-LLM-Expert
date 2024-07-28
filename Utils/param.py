
def model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    formatted_total_params = "{:,}".format(total_params)
    print(f"Total parameters: {formatted_total_params}")


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_trainable_params = "{:,}".format(trainable_params)
    print(f"Trainable parameters: {formatted_trainable_params}")


    non_trainable_params = total_params - trainable_params
    formatted_non_trainable_params = "{:,}".format(non_trainable_params)
    print(f"Non-trainable parameters: {formatted_non_trainable_params}")