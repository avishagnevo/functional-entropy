"""
An implementation of the paper: "Removing Bias in Multi-modal Classifiers: Regularization by Maximizing Functional
 Entropies" NeurIPS 2020.
"""

import torch
from typing import List  , Optional


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
        #torch.manual_seed(0)
        #return tens + torch.randn_like(tens) * tens.std(dim=over_dim)
        return tens + torch.randn_like(tens)

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
    
    def perturb_tensor_subset(
        tens: torch.Tensor,
        pixels: List[int],
        n_samples: int,
        perturbation: bool = True,
        num_channels: int = 3,
        over_dim: int = 0
        ) -> torch.Tensor:
        """
        :param tens: shape (B, C, H, W) float
        :param pixels: list of pixel indices in [0..(H*W)-1]
        :param n_samples: replicate factor (per pixel)
        :param perturbation: if True, add random noise
        :param num_channels: normally equals C
        :param over_dim: not used for scale, only for shape if needed
        :return: shape (len(pixels)*B*n_samples, C, H, W)
        """
        B, C, H, W = tens.shape
        F = C * H * W

        tens.requires_grad = False

        base_flat = tens.reshape(B, F)

        all_noisy = []
        base_mask = base_flat.new_zeros(F)
        jump = F // num_channels

        for pix in pixels:
            flat = base_flat.repeat(1, n_samples)

            flat = flat.view(B*n_samples, F)

            if perturbation:
                mask = base_mask.clone()

                for c_idx in range(num_channels):
                    idx = pix + c_idx * jump
                    mask[idx] = 1.0

                torch.manual_seed(0)
                noise = torch.randn_like(flat) * mask
                #noise = torch.normal(0.0,1.0, (1,1)).to(mask.device) * mask
                flat = flat + noise
                #flat = torch.clamp(flat, min=0.0, max=1.0) #why is tens values not between -1 and 1

            out = flat.view(B*n_samples, C, H, W)
            all_noisy.append(out)

        # cat => (len(pixels)*B*n_samples, C, H, W)
        pertube_tens = torch.cat(all_noisy, dim=0)
        return pertube_tens

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
    def get_image_importance_by_estimation(cls, grad: torch.Tensor, n_samples: int, estimation: str = 'var', softmax_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the mean squared norm of gradients for each pixel group.
        
        Assumes grad is of shape (B, C, H, W) with B = n_samples * num_pixels, where each contiguous 
        block of n_samples corresponds to a single pixel's perturbations.
        
        :param grad: Tensor of gradients with shape (B, C, H, W)
        :param n_samples: Number of perturbations per pixel.
        :param estimation: If 'var', return the per-group mean squared norm as a vector. if 'ent', return the
        per-group mean squared norm divided by the corresponding softmax probability. Otherwise, return the overall mean (scalar).
        :param softmax_prob: Tensor of shape (B,) with the softmax probabilities for each perturbation.
        :return: Tensor of shape (num_pixels,) if estimation=='var', else a scalar.
        """
        B, C, H, W = grad.shape
        num_pixels = B // n_samples  # number of groups (pixels)
        
        # Reshape to (num_pixels, n_samples, C, H, W)
        grad_grouped = grad.view(num_pixels, n_samples, C, H, W)
        
        # Compute the squared L2 norm for each sample in the group; result shape: (num_pixels, n_samples, H, W)
        group_norms_sq = torch.norm(grad_grouped, p=2, dim=2) ** 2
        ##
        #group_sum_sq = grad_grouped.sum(dim=(2,3,4))  # shape: (num_pixels,n_samples)
        #group_sum_sq = (grad_grouped**2).sum(dim=(2,3,4))  # shape: (num_pixels,n_samples)
        #group_sum_sq = grad_grouped.sum(dim=(1,2,3,4))  # shape: (num_pixels,n_samples)
        #group_sum_sq = (grad_grouped**2).sum(dim=(1,2,3,4))  # shape: (num_pixels,n_samples)
        ##
        
        if estimation == 'var':
            # Mean over the perturbations' squared norms
            mean_sq_norm = group_norms_sq.mean(dim=1)  # shape: (num_pixels,H, W)    
            return mean_sq_norm  # per-pixel vector of mean squared norms
        elif estimation == 'ent':
            # Devide sq_norm by correspoding softmax_prob, and then calculate the mean
            assert softmax_prob is not None, "softmax_prob must be provided for 'ent' estimation"
            # Normalize by the softmax probability for each group
            print('####')
            print(softmax_prob.shape)
            print(group_norms_sq.shape)
            print(softmax_prob.view(num_pixels, n_samples).shape)
            print(softmax_prob)
            group_norms_sq_normalized = group_norms_sq / softmax_prob.view(num_pixels, n_samples)
            # Mean over the perturbations for each pixel
            mean_sq_norm_normalized = group_norms_sq_normalized.mean(dim=1)  # shape: (num_pixels,H, W)
            return mean_sq_norm_normalized  # per-pixel vector of mean squared norms
        else:
            # Mean over all
            stop
            mean_sq_norm = group_norms_sq.mean(dim=1)  # shape: (num_pixels,)   v
            return mean_sq_norm.mean()  # overall scalar average
        
        
    @classmethod
    def get_importance_by_estimation(cls, grad: torch.Tensor, n_samples: int, estimation: str = 'var', softmax_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the mean squared norm of gradients for each pixel group.
        
        Assumes grad is of shape (B, C, H, W) with B = n_samples * num_pixels, where each contiguous 
        block of n_samples corresponds to a single pixel's perturbations.
        
        :param grad: Tensor of gradients with shape (B, C, H, W)
        :param n_samples: Number of perturbations per pixel.
        :param estimation: If 'var', return the per-group mean squared norm as a vector. if 'ent', return the
        per-group mean squared norm divided by the corresponding softmax probability. Otherwise, return the overall mean (scalar).
        :param softmax_prob: Tensor of shape (B,) with the softmax probabilities for each perturbation.
        :return: Tensor of shape (num_pixels,) if estimation=='var', else a scalar.
        """
        B, C, H, W = grad.shape
        num_pixels = B // n_samples  # number of groups (pixels)
        
        # Reshape to (num_pixels, n_samples, C, H, W)
        grad_grouped = grad.view(num_pixels, n_samples, C, H, W)
        
        # Compute the squared L2 norm for each sample in the group; result shape: (num_pixels, n_samples)
        group_norms_sq = torch.norm(grad_grouped, p=2, dim=(2, 3, 4)) ** 2
        ##
        #group_sum_sq = grad_grouped.sum(dim=(2,3,4))  # shape: (num_pixels,n_samples)
        #group_sum_sq = (grad_grouped**2).sum(dim=(2,3,4))  # shape: (num_pixels,n_samples)
        #group_sum_sq = grad_grouped.sum(dim=(1,2,3,4))  # shape: (num_pixels,n_samples)
        #group_sum_sq = (grad_grouped**2).sum(dim=(1,2,3,4))  # shape: (num_pixels,n_samples)
        ##
        
        if estimation == 'var':
            # Mean over the perturbations' squared norms
            mean_sq_norm = group_norms_sq.mean(dim=1)  # shape: (num_pixels,)    
            return mean_sq_norm  # per-pixel vector of mean squared norms
        elif estimation == 'ent':
            # Devide sq_norm by correspoding softmax_prob, and then calculate the mean
            assert softmax_prob is not None, "softmax_prob must be provided for 'ent' estimation"
            # Normalize by the softmax probability for each group
            group_norms_sq_normalized = group_norms_sq / softmax_prob.view(num_pixels, n_samples)
            # Mean over the perturbations for each pixel
            mean_sq_norm_normalized = group_norms_sq_normalized.mean(dim=1)  # shape: (num_pixels,)
            return mean_sq_norm_normalized  # per-pixel vector of mean squared norms
        else:
            # Mean over all
            stop
            mean_sq_norm = group_norms_sq.mean(dim=1)  # shape: (num_pixels,)   v
            return mean_sq_norm.mean()  # overall scalar average
    
    @classmethod
    def _get_importance_by_estimation(cls, grad: torch.Tensor, n_samples: int, 
                                                estimation: str = 'var', 
                                                softmax_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the importance of each pixel group by first computing the L2 norm per channel 
        over the spatial dimensions, averaging these norms over channels, then squaring that average,
        and finally averaging over n_samples perturbations.
        
        This produces a per-pixel importance value (vector of shape (num_pixels,)), which can be reshaped to (H, W).
        
        :param grad: Tensor of gradients with shape (B, C, H, W) where B = n_samples * num_pixels.
        :param n_samples: Number of perturbations per pixel.
        :param estimation: If 'var', returns the per-group importance as described. 
                        If 'ent', returns the per-group importance normalized by softmax_prob.
                        Otherwise, returns the overall mean as a scalar.
        :param softmax_prob: Tensor of shape (B,) with softmax probabilities for each perturbation 
                            (required if estimation == 'ent').
        :return: Tensor of shape (num_pixels,) if estimation is 'var' or 'ent', else a scalar.
        """
        B, C, H, W = grad.shape
        num_pixels = B // n_samples  # number of pixels
        
        # Reshape to (num_pixels, n_samples, C, H, W)
        grad_grouped = grad.view(num_pixels, n_samples, C, H, W)
        
        # Step 1: Compute the L2 norm per channel over the spatial dims (H, W)
        # This results in a tensor of shape (num_pixels, n_samples, C)
        channel_norms = torch.norm(grad_grouped, p=2, dim=(3, 4))
        
        # Step 2: Average the norms over the channels --> shape: (num_pixels, n_samples)
        mean_channel_norm = channel_norms.mean(dim=2)
        
        # Step 3: Square the averaged norm (per sample)
        group_importance = mean_channel_norm ** 2  # shape: (num_pixels, n_samples)
        
        # Step 4: Average over the n_samples for each pixel group
        if estimation == 'var':
            mean_importance = group_importance.mean(dim=1)  # shape: (num_pixels,)
            return mean_importance
        elif estimation == 'ent':
            assert softmax_prob is not None, "softmax_prob must be provided for 'ent' estimation"
            # Normalize by the softmax probability for each perturbation
            group_importance_normalized = group_importance / softmax_prob.view(num_pixels, n_samples)
            mean_importance_normalized = group_importance_normalized.mean(dim=1)
            return mean_importance_normalized
        else:
            # Return overall scalar average
            return group_importance.mean()
        

    @classmethod
    def get_importance_by_estimation_c(cls, grad: torch.Tensor, n_samples: int, 
                                    estimation: str = 'var', 
                                    softmax_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the importance of each pixel group by:
        1. Computing the L2 norm per channel over the spatial dimensions (for each perturbation sample).
        2. Squaring the per-channel norm.
        3. Averaging these squared values over the n_samples for each channel independently.
        4. Finally, averaging over the channels to produce a single importance value per pixel.
        
        This produces a per-pixel importance vector (of shape (num_pixels,)), which can be reshaped to (H, W).
        
        :param grad: Tensor of gradients with shape (B, C, H, W) where B = n_samples * num_pixels.
        :param n_samples: Number of perturbations per pixel.
        :param estimation: 
            - If 'var', returns the per-group importance as described.
            - If 'ent', returns the per-group importance normalized by softmax_prob.
            - Otherwise, returns the overall mean as a scalar.
        :param softmax_prob: Tensor of shape (B,) with softmax probabilities for each perturbation 
                            (required if estimation == 'ent').
        :return: Tensor of shape (num_pixels,) if estimation is 'var' or 'ent', else a scalar.
        """
        B, C, H, W = grad.shape
        num_pixels = B // n_samples  # total number of pixels
        
        # Reshape gradients to group the n_samples for each pixel:
        # Shape becomes (num_pixels, n_samples, C, H, W)
        grad_grouped = grad.view(num_pixels, n_samples, C, H, W)
        
        # Step 1: Compute the L2 norm over the spatial dimensions for each channel.
        # This yields a tensor of shape (num_pixels, n_samples, C)
        channel_norms = torch.norm(grad_grouped, p=2, dim=(3, 4))
        
        # Step 2: Square the per-channel norms.
        channel_norms_sq = channel_norms ** 2  # shape: (num_pixels, n_samples, C)
        
        # Step 3: Average over the n_samples for each pixel and channel independently.
        # This results in a tensor of shape (num_pixels, C)
        per_channel_importance = channel_norms_sq.mean(dim=1)
        
        # If 'ent' estimation, normalize by the softmax probability for each perturbation first.
        if estimation == 'ent':
            assert softmax_prob is not None, "softmax_prob must be provided for 'ent' estimation"
            # Reshape softmax_prob to (num_pixels, n_samples)
            softmax_prob_grouped = softmax_prob.view(num_pixels, n_samples)
            # Divide each squared norm by the corresponding softmax probability.
            channel_norms_sq_normalized = channel_norms_sq / softmax_prob_grouped.unsqueeze(2)
            per_channel_importance = channel_norms_sq_normalized.mean(dim=1)
        
        # Step 4: Finally, average over the channels (aggregate last) to get a scalar per pixel.
        final_importance = per_channel_importance.mean(dim=1)  # shape: (num_pixels,)
        
        if estimation not in ['var', 'ent']:
            return final_importance.mean()  # overall scalar average
        else:
            return final_importance
        


    
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
        self.c = 0.5
