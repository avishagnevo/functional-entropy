"""
An implementation of the paper: "Removing Bias in Multi-modal Classifiers: Regularization by Maximizing Functional
 Entropies" NeurIPS 2020.
"""

import torch
from typing import List  


class Perturbation:
    """
    Class that in charge of the perturbation techniques
    """
    @classmethod
    def _add_noise_to_tensor(cls, tens: torch.Tensor, over_dim: int = 0) -> torch.Tensor:
        """
        Adds noise to a tensor sampled from N(0, tens.std()).
        :param tens:
        :param over_dim: over what dim to calculate the std. 0 for features over batch,  1 for over sample.
        :return: noisy tensor in the same shape as input
        """

        return tens + torch.randn_like(tens) * tens.std(dim=over_dim)
        # return tens + torch.randn_like(tens)

    @classmethod
    def _add_noise_to_tensor_exept_pixel(cls, tens: torch.Tensor, pixel: int , num_channels: int=3, over_dim: int = 0) -> torch.Tensor:
        """
        Adds noise to a tensor sampled from N(0, tens.std()).
        :param tens:
        :param pixel: pixel index to exclude from perturbation
        :param num_channels: number of channels in the tensor
        :param over_dim: over what dim to calculate the std. 0 for features over batch,  1 for over sample.
        :return: noisy tensor in the same shape as input
        """
        pertubated_tens = tens.add_(torch.randn_like(tens) * torch.ones_like(tens.std(dim=over_dim)))
        jump = tens.shape[-1]//num_channels

        for channel in range(num_channels):
            pertubated_tens[:, pixel + channel * jump] = tens[:, pixel + channel * jump]

        return pertubated_tens  

    @classmethod
    def add_noise_to_tensor_subset(cls, tens: torch.Tensor, pixels: List[int], n_samples: int, 
                                    num_channels: int = 3, over_dim: int = 0) -> torch.Tensor:
        """
        Adds noise to only a subset of positions (the given pixels, applied to all channels)
        in a batched tensor. Assumes tens is a flattened tensor of shape (B * n_samples * L, F),
        where F = num_channels * (H * W) and L = len(pixels).
        
        For each original sample, the ordering of tens is assumed to be such that the first n_samples
        rows correspond to perturbations for pixels[0], the next n_samples for pixels[1], etc.
        
        :param tens: tensor of shape (B * n_samples * L, F)
        :param pixels: list of pixel indices (spatial indices)
        :param n_samples: number of perturbations per pixel.
        :param num_channels: number of channels (e.g., 3)
        :param over_dim: (unused here, kept for compatibility)
        :return: tens with noise added only at the specified positions.
        """
        BnL, F = tens.shape
        HW = F // num_channels
        L = len(pixels)
        # Infer original batch size: B = total_rows / (n_samples * L)
        B = BnL // (n_samples * L)
        
        # Create a base mask for each pixel in 'pixels'.
        # For a given pixel p (a spatial index in [0, H*W)),
        # we set to 1 the positions p + c * HW for c in 0...num_channels-1.
        masks = []
        for p in pixels:
            mask = torch.zeros(F, device=tens.device, dtype=tens.dtype)
            for c in range(num_channels):
                index = p + c * HW
                mask[index] = 1.0
            masks.append(mask)
        # Shape: (L, F)
        base_mask = torch.stack(masks, dim=0)
        
        # For each pixel, repeat its mask n_samples times. This gives a per-sample mask of shape (L*n_samples, F)
        mask_per_sample = base_mask.repeat_interleave(n_samples, dim=0)  # shape: (L * n_samples, F)
        
        # Tile this mask for each original sample in the batch.
        mask_total = mask_per_sample.repeat(B, 1)  # shape: (B * L * n_samples, F)
        
        # Generate noise: here we use tens.std() as a scale.
        scale = 1  # scalar noise scale tens.std()
        noise = torch.randn_like(tens) * scale
        
        # Only add noise where mask_total is 1.
        perturbed_tens = tens + noise * mask_total

        return perturbed_tens

    @classmethod
    def _add_noise_to_tensor_subset(cls, tens: torch.Tensor, pixels : List[int] , num_channels: int=3, over_dim: int = 0) -> torch.Tensor:
        """
        Adds noise to a tensor sampled from N(0, tens.std()).
        :param tens:
        :param pixels: pixels index to include in perturbation
        :param num_channels: number of channels in the tensor
        :param over_dim: over what dim to calculate the std. 0 for features over batch,  1 for over sample.
        :return: noisy tensor in the same shape as input
        """
        pertube_flags = torch.zeros_like(tens.std(dim=over_dim))
        jump = tens.shape[-1]//num_channels

        for pixel in pixels:
            for channel in range(num_channels):
                pertube_flags[pixel + channel * jump] = 1

        pertubated_tens = tens.clone() + torch.randn_like(tens) * pertube_flags

        '''
        for pixel in pixels:
            for channel in range(num_channels):
                print('!=', pertubated_tens[:, pixel + channel * jump])
                print('!=',tens[:, pixel + channel * jump])
                print('=', pertubated_tens[:, pixel + channel * jump +1])
                print('=',tens[:, pixel + channel * jump +1])
        '''        
        
        return pertubated_tens      

    @classmethod
    def perturb_tensor(cls, tens: torch.Tensor, n_samples: int, perturbation: bool = True) -> torch.Tensor:
        """
        Flatting the tensor, expanding it, perturbing and reconstructing to the original shape.
        Note, this function assumes that the batch is the first dimension.
        :param tens:
        :param n_samples: times to perturb
        :param perturbation: False - only duplicating the tensor
        :return: tensor in the shape of [batch, samples * num_eval_samples]
        """
        tens_dim = list(tens.shape)

        #tens = tens.view(tens.shape[0], -1)
        tens = tens.reshape(tens.shape[0], -1)
        tens = tens.repeat(1, n_samples)

        tens = tens.view(tens.shape[0] * n_samples, -1)

        if perturbation:
            tens = cls._add_noise_to_tensor(tens)

        tens_dim[0] *= n_samples

        tens = tens.view(*tens_dim)
        tens.requires_grad_()

        return tens

    @classmethod
    def perturb_tensor_exept_pixel(cls, tens: torch.Tensor, pixel: int, n_samples: int, perturbation: bool = True) -> torch.Tensor:
        """
        Flatting the tensor, expanding it, perturbing exept from pixel and reconstructing to the original shape.
        Note, this function assumes that the batch is the first dimension.
        :param tens:
        :param pixel: pixel index to exclude from perturbation
        :param n_samples: times to perturb
        :param perturbation: False - only duplicating the tensor
        :return: tensor in the shape of [batch, samples * num_eval_samples]
        """
        '''
        tens = torch.zeros_like(tens.clone())
        ones = torch.ones_like(tens[:,:,pixel,pixel].clone())
        tens[:,:,pixel,pixel] = ones
        '''

        tens_dim = list(tens.shape)
        max_pixel = tens_dim[-2]*tens_dim[-1]
        num_channels=tens_dim[1]

        assert pixel < max_pixel, f"Pixel index {pixel} is out of range 0:{max_pixel}"

        tens = tens.view(tens.shape[0], -1)
        tens = tens.repeat(1, n_samples)

        tens = tens.view(tens.shape[0] * n_samples, -1)

        if perturbation:
            tens = cls._add_noise_to_tensor_exept_pixel(tens, pixel, num_channels)

        tens_dim[0] *= n_samples

        tens = tens.view(*tens_dim)
        tens.requires_grad_()

        return tens    

    @classmethod
    def _perturb_tensor_subset(cls, tens: torch.Tensor, pixels: List[int], n_samples: int, perturbation: bool = True) -> torch.Tensor:
        """
        Flatting the tensor, expanding it, perturbing exept from pixel and reconstructing to the original shape.
        Note, this function assumes that the batch is the first dimension.
        :param tens:
        :param pixels: pixels index to include in perturbation
        :param n_samples: times to perturb
        :param perturbation: False - only duplicating the tensor
        :return: tensor in the shape of [batch, samples * num_eval_samples]
        """
        '''
        tens = torch.zeros_like(tens.clone())
        pixel=pixels[0]
        ones = torch.ones_like(tens[:,:,pixel,pixel].clone())
        tens[:,:,pixel,pixel] = ones
        print('tens.shape', tens.shape)'''
        
        tens_dim = list(tens.shape)
        max_pixel = tens_dim[-2]*tens_dim[-1]
        num_channels=tens_dim[1]

        for pixel in pixels:
            assert pixel < max_pixel, f"Pixel index {pixel} is out of range 0:{max_pixel}"
            
        #print('tens.shape', tens.shape)    

        #tens = tens.view(tens.shape[0], -1)
        #tens = tens.reshape(tens.shape[0], -1)
        tens = tens.contiguous().view(tens.shape[0], -1)
        tens = tens.repeat(1, n_samples)

        tens = tens.view(tens.shape[0] * n_samples, -1)

        if perturbation:
            tens = cls._add_noise_to_tensor_subset(tens, pixels, num_channels)

        tens_dim[0] *= n_samples

        tens = tens.view(*tens_dim)
        tens.requires_grad_()

        return tens        
    

    @classmethod
    def perturb_tensor_subset(cls, tens: torch.Tensor, pixels: List[int], n_samples: int, perturbation: bool = True) -> torch.Tensor:
        """
        Flattens the tensor, expands it to have n_samples * len(pixels) copies per original sample,
        and perturbs (adds noise to) only the specified pixels in parallel.
        
        :param tens: input tensor of shape (batch, channels, H, W)
        :param pixels: list of pixel indices (spatial indices in range [0, H*W))
        :param n_samples: number of perturbations per pixel.
        :param perturbation: if False, only duplicates the tensor.
        :return: tensor of shape (batch * n_samples * len(pixels), channels, H, W) with perturbations.
        """
        """
        tens = torch.zeros_like(tens.clone())
        pixel=pixels[0]
        ones = torch.ones_like(tens[:,:,pixel,pixel].clone())
        tens[:,:,pixel,pixel] = ones
        print('tens.shape', tens.shape) 
        """

        B, C, H, W = tens.shape
        F = C * H * W
        max_pixel = H * W
        for pixel in pixels:
            assert pixel < max_pixel, f"Pixel index {pixel} is out of range 0:{max_pixel}"
        
        # Flatten tens to shape (B, F)
        tens_flat = tens.contiguous().view(B, -1)
        total_repeats = n_samples * len(pixels)
        # Repeat each sample total_repeats times along the batch dimension.
        tens_expanded = tens_flat.repeat_interleave(total_repeats, dim=0)  # shape: (B * total_repeats, F)
        
        if perturbation:
            tens_expanded = cls.add_noise_to_tensor_subset(tens_expanded, pixels, n_samples, num_channels=C)
        
        # Reshape back to (B * total_repeats, C, H, W)
        tens_out = tens_expanded.view(B * total_repeats, C, H, W)
        tens_out.requires_grad_()

        return tens_out

    @classmethod
    def get_expanded_logits(cls, logits: torch.Tensor, n_samples: int, logits_flg: bool = True) -> torch.Tensor:
        """
        Perform Softmax and then expand the logits depends on the num_eval_samples
        :param logits_flg: whether the input is logits or softmax
        :param logits: tensor holds logits outputs from the model
        :param n_samples: times to duplicate
        :return:
        """
        if logits_flg:
            logits = torch.nn.functional.softmax(logits, dim=1)
        expanded_logits = logits.repeat(1, n_samples)

        return expanded_logits.view(expanded_logits.shape[0] * n_samples, -1)


class Regularization(object):
    """
    Class that in charge of the regularization techniques
    """
    @classmethod
    def _get_variance_loss(cls, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance along samples for the first dimension in a tensor
        :param loss: [batch, number of evaluate samples]
        :return: variance of a given batch of loss values
        """

        return torch.var(loss, dim=1)

    @classmethod
    def _get_variance(cls, softmaxs: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance along samples for the first dimension in a tensor
        :param softmaxs: [batch, number of evaluate samples]
        :return: variance of a given batch of softmax values
        """

        return torch.var(softmaxs, dim=1)    

    @classmethod
    def _get_differential_entropy(cls, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes differential entropy: -E[flogf]
        :param loss:
        :return: a tensor holds the differential entropy for a batch
        """

        return -1 * torch.sum(loss * loss.log())

    @classmethod
    def _get_functional_entropy(cls, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes functional entropy: E[flogf] - E[f]logE[f]
        :param loss:
        :return: a tensor holds the functional entropy for a batch
        """
        loss = torch.nn.functional.normalize(loss, p=1, dim=1)
        loss = torch.mean(loss * loss.log()) - (torch.mean(loss) * torch.mean(loss).log())

        return loss

    @classmethod
    def get_batch_statistics(cls, loss: torch.Tensor, n_samples: int, estimation: str = 'ent') -> torch.Tensor:
        """
        Calculate the expectation of the batch gradient
        :param n_samples:
        :param loss:
        :param estimation:
        :return: Influence expectation
        """
        loss = loss.reshape(-1, n_samples)

        if estimation == 'var':
            batch_statistics = cls._get_variance(loss)
            batch_statistics = torch.abs(batch_statistics)
        elif estimation == 'ent':
            batch_statistics = cls._get_functional_entropy(loss)
        elif estimation == 'dif_ent':
            batch_statistics = cls._get_differential_entropy(loss)
        else:
            raise NotImplementedError(f'{estimation} is unknown regularization, please use "var" or "ent".')

        return torch.mean(batch_statistics)

    @classmethod
    def get_grad_sqrd_norm_mean(cls, grad: torch.Tensor, n_samples: int, estimation: str = 'var') -> torch.Tensor:
        """
        Calculates the mean squared norm of gradients for each pixel group.
        
        Assumes grad is of shape (B, C, H, W) with B = n_samples * num_pixels, where each contiguous 
        block of n_samples corresponds to a single pixel's perturbations.
        
        :param grad: Tensor of gradients with shape (B, C, H, W)
        :param n_samples: Number of perturbations per pixel.
        :param estimation: If 'var', return the per-group mean squared norm as a vector.
                        Otherwise, return the overall mean (scalar).
        :return: Tensor of shape (num_pixels,) if estimation=='var', else a scalar.
        """
        B, C, H, W = grad.shape
        num_pixels = B // n_samples  # number of groups (pixels)
        
        # Reshape to (num_pixels, n_samples, C, H, W)
        grad_grouped = grad.view(num_pixels, n_samples, C, H, W)
        
        # Compute the squared L2 norm for each sample in the group; result shape: (num_pixels, n_samples)
        group_norms_sq = torch.norm(grad_grouped, p=2, dim=(2, 3, 4)) ** 2
        
        # Mean over the perturbations for each pixel
        mean_sq_norm = group_norms_sq.mean(dim=1)  # shape: (num_pixels,)
        
        if estimation == 'var':
            return mean_sq_norm  # per-pixel vector of mean squared norms
        else:
            return mean_sq_norm.mean()  # overall scalar average

    
    @classmethod
    def get_batch_norm(cls, grad: torch.Tensor, loss: torch.Tensor = None, estimation: str = 'ent') -> torch.Tensor:
        """
        Calculate the expectation of the batch gradient norms squered
        :param loss:
        :param estimation:
        :param grad: tensor holds the gradient batch
        :return: approximation of the required expectation
        """
        batch_grad_norm = torch.norm(grad, p=2, dim=1)
        batch_grad_norm = torch.pow(batch_grad_norm, 2)

        if estimation == 'ent':
            batch_grad_norm = batch_grad_norm / loss

        return torch.mean(batch_grad_norm)

    @classmethod
    def _get_batch_norm(cls, grad: torch.Tensor, loss: torch.Tensor = None, estimation: str = 'ent') -> torch.Tensor:
        """
        Calculate the expectation of the batch gradient
        :param loss:
        :param estimation:
        :param grad: tensor holds the gradient batch
        :return: approximation of the required expectation
        """
        batch_grad_norm = torch.norm(grad, p=2, dim=1)
        batch_grad_norm = torch.pow(batch_grad_norm, 2)

        if estimation == 'ent':
            batch_grad_norm = batch_grad_norm / loss

        return batch_grad_norm

    @classmethod
    def _get_max_ent(cls, inf_scores: torch.Tensor, norm: float) -> torch.Tensor:
        """
        Calculate the norm of 1 divided by the information
        :param inf_scores: tensor holding batch information scores
        :param norm: which norm to use
        :return:
        """
        return torch.norm(torch.div(1, inf_scores), p=norm)

    @classmethod
    def _get_max_ent_minus(cls, inf_scores: torch.Tensor, norm: float) -> torch.Tensor:
        """
        Calculate -1 * the norm of the information
        :param inf_scores: tensor holding batch information scores
        :param norm: which norm to use
        :return:
        """
        return -1 * torch.norm(inf_scores, p=norm) + 0.1

    @classmethod
    def get_regularization_term(cls, inf_scores: torch.Tensor, norm: float = 2.0,
                                optim_method: str = 'max_ent') -> torch.Tensor:
        """
        Compute the regularization term given a batch of information scores
        :param inf_scores: tensor holding a batch of information scores
        :param norm: defines which norm to use (1 or 2)
        :param optim_method: Define optimization method (possible methods: "min_ent", "max_ent", "max_ent_minus",
         "normalized")
        :return:
        """

        if optim_method == 'max_ent':
            return cls._get_max_ent(inf_scores, norm)
        elif optim_method == 'min_ent':
            return torch.norm(inf_scores, p=norm)
        elif optim_method == 'max_ent_minus':
            return cls._get_max_ent_minus(inf_scores, norm)

        raise NotImplementedError(f'"{optim_method}" is unknown')


class RegParameters(object):
    """
    This class controls all the regularization-related properties
    """
    def __init__(self, lambda_: float = 1e-10, norm: float = 2.0, estimation: str = 'ent',
                 optim_method: str = 'max_ent', n_samples: int = 25, grad: bool = True):
        self.lambda_ = lambda_
        self.norm = norm
        self.estimation = estimation
        self.optim_method = optim_method
        self.n_samples = n_samples
        self.grad = grad
