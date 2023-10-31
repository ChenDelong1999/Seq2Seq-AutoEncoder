import torch

mse = torch.nn.MSELoss()

def seq2seq_autoencoder_loss(prediction, traget, channel_info):
    loss = {}
    for name, dim in channel_info.items():
        if type(dim) != list:
            dim_start, dim_end = dim, dim+1
        else:
            dim_start, dim_end = dim
        # prediction_loss = self.loss(prediction[:, :-1, :], data[:, 1:, :])
        loss[name] = mse(prediction[:, :-1, dim_start:dim_end], traget[:, 1:, dim_start:dim_end])

    return loss