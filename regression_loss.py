import torch


class MyRegressionLoss:
    """
    Implements the loss function for the localization task
        formulated as regression loss in combination with two binary cross entropies.
    The weighting factors of the different loss terms can be customized via the constructor.
    """
    def __init__(self, bce_weights, ab_factor=2.5):
        self.bce_weights = bce_weights
        self.ab_weighting_factor = ab_factor

    def calc_metric(self, predictions, labels):
        bce = torch.nn.BCEWithLogitsLoss()
        bc_tv = bce(predictions[:, 1], labels[:, 1])
        bc_tm = bce(predictions[:, 2], labels[:, 2])

        mse = torch.nn.MSELoss()
        mse_ab = mse(self.ab_weighting_factor * predictions[:, 0], self.ab_weighting_factor * labels[:, 0])
        mse_rt_sin = mse(torch.sqrt(labels[:, 0]) * predictions[:, 3], torch.sqrt(labels[:, 0]) * labels[:, 3])
        mse_rt_cos = mse(torch.sqrt(labels[:, 0]) * predictions[:, 4], torch.sqrt(labels[:, 0]) * labels[:, 4])

        return self.bce_weights[0] * bc_tv + self.bce_weights[1] * bc_tm + mse_ab + mse_rt_sin + mse_rt_cos
