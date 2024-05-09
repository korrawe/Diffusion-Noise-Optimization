import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_loaders.humanml.data.dataset import HumanML3D
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.utils.metrics import *


def calculate_fid_given_two_populations(
    motion_a,
    motion_b,
    len_a,
    len_b,
    dataset: HumanML3D,
    dataset_name="humanml",
    device="cuda",
    batch_size=64,
):
    """
    Args:
        motion_a: [N, feature_dim, 1, max_len]
        motion_b: [N, feature_dim, 1, max_len]
        len_a: [N]
        len_b: [N]
    """
    eval_wrapper = EvaluatorMDMWrapper(dataset_name, device)

    def reshape_motion(motion):
        # [N, feature, 1, max_len] => [N, max_len, feature_dim]
        motion = motion.squeeze().permute(0, 2, 1).cpu().numpy()
        return motion

    # [N, max_len, feature_dim]
    motion_a = reshape_motion(motion_a)
    motion_b = reshape_motion(motion_b)

    def renorm_motion(motion_i):
        """
        Args:
            motion_i: [len, feat]
        """
        # [len, feat]
        denormed_motion = dataset.t2m_dataset.inv_transform(motion_i)
        # [len, feat]
        renormed_motion = (
            denormed_motion - dataset.mean_for_eval
        ) / dataset.std_for_eval  # according to T2M norms
        return renormed_motion

    # renorm motion one by one
    # [N, max_len, feature_dim]
    motion_a = np.stack([renorm_motion(motion_i) for motion_i in motion_a], axis=0)
    motion_b = np.stack([renorm_motion(motion_i) for motion_i in motion_b], axis=0)

    motion_a = torch.tensor(motion_a)
    motion_b = torch.tensor(motion_b)

    def get_embeddding(motion, motion_len, desc):
        motion_dataset = TensorDataset(motion, motion_len)
        loader = DataLoader(motion_dataset, batch_size=batch_size, shuffle=False)
        # apply eval_wrapper to get motion embeddings
        motion_embeddings = []
        for _motion, _len in tqdm(loader, desc=desc):
            _motion = _motion.to(device)
            # [batch_size, feature_dim]
            motion_emb = (
                eval_wrapper.get_motion_embeddings(motions=_motion, m_lens=_len)
                .detach()
                .cpu()
                .numpy()
            )
            motion_embeddings.append(motion_emb)
        # [all, feature_dim]
        motion_embeddings = np.concatenate(motion_embeddings, axis=0)
        return motion_embeddings

    # [all, feature_dim]
    motion_a = get_embeddding(motion_a, len_a, desc="motion_a")
    motion_b = get_embeddding(motion_b, len_b, desc="motion_b")

    # [feature dim]
    mu_a, cov_a = calculate_activation_statistics(motion_a)
    mu_b, cov_b = calculate_activation_statistics(motion_b)
    fid = calculate_frechet_distance(mu_a, cov_a, mu_b, cov_b)
    return fid
