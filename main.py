from typing import Any, Optional, Sequence
import argparse
import os

import random
import numpy as np
from nptyping import NDArray, Float, Shape
import pandas as pd
import torch
from torch import Tensor
from torch.nn.modules import Module
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE
from scipy.spatial import distance

import utils
from model import Model
from datasets import Chesapeake_CIFAR10


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

use_cuda = torch.cuda.is_available()
device = torch.device("gpu:0" if use_cuda else "cpu")


def off_diagonal(x: Tensor) -> Tensor:
    """Return a flattened view of the off-diagonal elements of a square matrix.

    Args:
        x (Tensor): Tensor to flatten.

    Returns:
        Tensor: Flattened view of the off-diagonal elements of input matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train(net: Module, data_loader: DataLoader, train_optimizer) -> float:
    """Train for one epoch to learn unique features.

    Args:
        net (Module): Network to train.
        data_loader (DataLoader): Dataloader of training data.
        train_optimizer: Optimiser for training.

    Returns:
        float: Average training loss.
    """
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_tuple in train_bar:
        (pos_1, pos_2), _ = data_tuple
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)
        # Barlow Twins.

        # Normalize the representations along the batch dimension.
        out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
        out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)

        # Cross-correlation matrix
        c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

        # Loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        if corr_neg_one is False:
            # The loss described in the original Barlow Twin's paper
            # encouraging off_diag to be zero.
            off_diag = off_diagonal(c).pow_(2).sum()
        else:
            # Inspired by HSIC.
            # Encouraging off_diag to be negative ones.
            off_diag = off_diagonal(c).add_(1).pow_(2).sum()
        loss = on_diag + lmbda * off_diag

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        if corr_neg_one is True:
            off_corr = -1
        else:
            off_corr = 0
        train_bar.set_description(
            "Train Epoch: [{}/{}] Loss: {:.4f} off_corr:{} lmbda:{:.4f} bsz:{} f_dim:{} dataset: {}".format(
                epoch,
                epochs,
                total_loss / total_num,
                off_corr,
                lmbda,
                batch_size,
                feature_dim,
                dataset,
            )
        )
    return total_loss / total_num


def test(
    net: Module, memory_data_loader: DataLoader, test_data_loader: DataLoader
) -> tuple[float, float]:
    """Test for one epoch, use weighted knn to find the most similar images' label to assign the test image.

    Args:
        net (Module): Network to test.
        memory_data_loader (DataLoader): _description_
        test_data_loader (DataLoader): _description_

    Returns:
        tuple[float, float]: Percentage top-1 and top-5 test accuracies.
    """
    net.eval()

    total_top1, total_top5, total_num, _feature_bank, target_bank = 0.0, 0.0, 0, [], []

    with torch.no_grad():
        # Generate feature bank and target bank.
        for data_tuple in tqdm(memory_data_loader, desc="Feature extracting"):
            (data, _), target = data_tuple
            target_bank.append(target)
            feature, _ = net(data.cuda(non_blocking=True))
            _feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(_feature_bank, dim=0).t().contiguous()

        # [N]
        feature_labels = (
            torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        )

        # Loop test data to predict the label by weighted knn search.
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, _ = net(data)

            total_num += data.size(0)

            # Compute cos similarity between each feature vector and feature bank ---> [B, N].
            sim_matrix = torch.mm(feature, feature_bank)

            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)

            # [B, K]
            sim_labels = torch.gather(
                feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices
            )
            sim_weight = (sim_weight / temperature).exp()

            # Counts for each class.
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)

            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0
            )

            # Weighted score ---> [B, C].
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1),
                dim=1,
            )

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            test_bar.set_description(
                "Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
                    epoch,
                    epochs,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100,
                )
            )

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def stack_rgb(
    image: NDArray[Shape["3, *, *"], Float],
    rgb: dict[str, int] = {"R": 0, "G": 1, "B": 2},
    max_value: int = 255,
) -> NDArray[Shape["*, *, 3"], Float]:
    """Stacks together red, green and blue image bands to create a RGB array.

    Args:
        image (NDArray[Shape["3, *, *"], Float]): Image of separate channels to be normalised
            and reshaped into stacked RGB image.
        rgb (Dict[str, int]): Optional; Dictionary of which channels in image are the R, G & B bands.
        max_value (int): Optional; The maximum pixel value in ``image``. e.g. for 8 bit this will be 255.

    Returns:
        NDArray[Shape["*, *, 3"], Float]: Normalised and stacked red, green, blue arrays into RGB array
    """

    # Extract R, G, B bands from image and normalise.
    channels: list[Any] = []
    for channel in ["R", "G", "B"]:
        band = image[rgb[channel]] / max_value
        channels.append(band)

    # Stack together RGB bands.
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays.
    rgb_image: NDArray[Shape["3, *, *"], Any] = np.dstack(
        (channels[2], channels[1], channels[0])
    )
    assert isinstance(rgb_image, np.ndarray)
    return rgb_image


def calc_norm_euc_dist(a: Sequence[int], b: Sequence[int]) -> float:
    """Calculates the normalised Euclidean distance between two vectors.

    Args:
        a (Sequence[int]): Vector `A`.
        b (Sequence[int]): Vector `B`.

    Returns:
        float: Normalised Euclidean distance between vectors `A` and `B`.
    """
    assert len(a) == len(b)
    euc_dist: float = distance.euclidean(a, b) / len(a)

    assert type(euc_dist) is float
    return euc_dist


def calc_avg_euc_dist(a: Tensor, b: Tensor) -> float:
    euc_dists = []
    for i in range(len(a)):
        euc_dists.append(
            calc_norm_euc_dist(a[i].detach().numpy(), b[i].detach().numpy())
        )

    return sum(euc_dists) / len(euc_dists)


def euc_dist_test(model: Module, loader) -> float:
    avg_euc_dist = 0.0
    model.eval()

    test_bar = tqdm(loader, total=100)
    i = 0
    for data_tuple in test_bar:
        i += 1
        if i > 100:
            continue
        else:
            (a, b), _ = data_tuple
            a, b = a.to(device), b.to(device)

            _, out_1 = model(a)
            _, out_2 = model(b)

            avg_euc_dist += calc_avg_euc_dist(out_1, out_2)

            test_bar.set_description(
                f"Step [{i}/{100}]: Average Euclidean Distance: {avg_euc_dist}"
            )

    avg_euc_dist = avg_euc_dist / len(loader)

    print(f"Average Euclidean Distance: {avg_euc_dist}")

    return avg_euc_dist


def tsne_cluster(
    model: Module,
    test_loader: DataLoader,
    n_dim: int = 2,
    filename: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    save: bool = True,
) -> None:
    """Perform TSNE clustering on the embeddings from the model and visualise."""
    image_pair, targets = next(iter(test_loader))

    images = image_pair[0]

    targets = targets.detach().numpy()

    model.eval()

    embeddings: Tensor = model.f(images.to(device))

    embeddings = embeddings.flatten(start_dim=1).detach().numpy()

    tsne = TSNE(n_dim, learning_rate="auto", init="random")

    x = tsne.fit_transform(embeddings)

    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(len(x)):
        plt.text(
            x[i, 0],
            x[i, 1],
            str(targets[i]),
            color=plt.cm.Set1(targets[i] / 10.0),
            fontdict={"weight": "bold", "size": 9},
        )

    images = images.detach().numpy()
    images = [stack_rgb(image, max_value=1400) for image in images]

    if hasattr(offsetbox, "AnnotationBbox"):
        # Only print thumbnails with matplotlib > 1.0.
        shown_images: NDArray[Any, Any] = np.array([[1.0, 1.0]])  # Just something big.

        for i in range(len(images)):
            dist = np.sum((x[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # Don’t show points that are too close.
                continue

            shown_images = np.r_[shown_images, [x[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
                x[i],
                frameon=False,
            )

            ax.add_artist(imagebox)

    # Hides the axes.
    plt.axis("off")

    if title is not None:
        plt.title(title)

    # Shows and/or saves plot.
    if show:
        plt.show()
    if save:
        plt.savefig(filename, bbox_inches="tight")
        print("TSNE cluster visualisation SAVED")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimCLR")
    parser.add_argument("-r", default="data", type=str, help="Root to dataset")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        help="Choose dataset from: chesapeake_cifar10, cifar10, tiny_imagenet or stl10",
    )
    parser.add_argument(
        "--feature_dim", default=128, type=int, help="Feature dim for latent vector"
    )
    parser.add_argument(
        "--temperature", default=0.5, type=float, help="Temperature used in softmax"
    )
    parser.add_argument(
        "--k",
        default=200,
        type=int,
        help="Top k most similar images used to predict the label",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        help="Number of sweeps over the dataset to train",
    )
    # For Barlow Twins.

    parser.add_argument(
        "--lmbda",
        default=0.005,
        type=float,
        help="Lambda that controls the on- and off-diagonal terms",
    )
    parser.add_argument(
        "--save_name_pre",
        default=None,
        type=str,
        help="Prefix for the filename of saved files. Override to load a specific saved model weights.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed number for replication of results.",
    )

    parser.add_argument("--corr_neg_one", dest="corr_neg_one", action="store_true")
    parser.add_argument("--corr_zero", dest="corr_neg_one", action="store_false")
    parser.set_defaults(corr_neg_one=False)

    parser.add_argument("--cluster_vis", dest="cluster_vis", action="store_true")
    parser.add_argument("--euc_dist", dest="euc_dist", action="store_true")

    # Args parse.
    args = parser.parse_args()
    dataset = args.dataset
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    lmbda = args.lmbda
    corr_neg_one = args.corr_neg_one

    cluster_vis = args.cluster_vis
    calc_euc_dist = args.euc_dist

    save_name_pre = args.save_name_pre

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)

    # Data prepare.
    if dataset == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root=args.r,
            train=True,
            transform=utils.CifarPairTransform(train_transform=True),
            download=True,
        )
        memory_data = torchvision.datasets.CIFAR10(
            root=args.r,
            train=True,
            transform=utils.CifarPairTransform(train_transform=False),
            download=True,
        )
        test_data = torchvision.datasets.CIFAR10(
            root=args.r,
            train=False,
            transform=utils.CifarPairTransform(train_transform=False),
            download=True,
        )
    if dataset == "chesapeake_cifar10":
        train_data = Chesapeake_CIFAR10(
            root=args.r,
            train=True,
            transform=utils.ChesapeakeCifarPairTransform(train_transform=True),
        )
        memory_data = Chesapeake_CIFAR10(
            root=args.r,
            train=True,
            transform=utils.ChesapeakeCifarPairTransform(train_transform=False),
        )
        test_data = Chesapeake_CIFAR10(
            root=args.r,
            train=False,
            transform=utils.ChesapeakeCifarPairTransform(train_transform=False),
        )
    elif dataset == "stl10":
        train_data = torchvision.datasets.STL10(
            root=args.r,
            split="train+unlabeled",
            transform=utils.StlPairTransform(train_transform=True),
            download=True,
        )
        memory_data = torchvision.datasets.STL10(
            root=args.r,
            split="train",
            transform=utils.StlPairTransform(train_transform=False),
            download=True,
        )
        test_data = torchvision.datasets.STL10(
            root=args.r,
            split="test",
            transform=utils.StlPairTransform(train_transform=False),
            download=True,
        )
    elif dataset == "tiny_imagenet":
        train_data = torchvision.datasets.ImageFolder(
            os.path.join(args.r, "tiny-imagenet-200", "train"),
            utils.TinyImageNetPairTransform(train_transform=True),
        )
        memory_data = torchvision.datasets.ImageFolder(
            os.path.join(args.r, "tiny-imagenet-200", "train"),
            utils.TinyImageNetPairTransform(train_transform=False),
        )
        test_data = torchvision.datasets.ImageFolder(
            os.path.join(args.r, "tiny-imagenet-200", "val"),
            utils.TinyImageNetPairTransform(train_transform=False),
        )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True
    )

    # Model setup and optimizer config.
    model = Model(feature_dim, dataset).to(device)
    if dataset == "cifar10":
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
    if dataset == "chesapeake_cifar10":
        flops, params = profile(model, inputs=(torch.randn(1, 4, 32, 32).to(device),))
    elif dataset == "tiny_imagenet" or dataset == "stl10":
        flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64).to(device),))

    flops, params = clever_format([flops, params])

    print("# Model Params: {} FLOPs: {}".format(params, flops))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    # Training loop.
    results: dict[str, list[float]] = {
        "train_loss": [],
        "test_acc@1": [],
        "test_acc@5": [],
    }
    if corr_neg_one is True:
        corr_neg_one_str = "neg_corr_"
    else:
        corr_neg_one_str = ""

    if save_name_pre is None:
        save_name_pre = (
            f"{corr_neg_one_str}{lmbda}_{feature_dim}_{batch_size}_{dataset}"
        )

    if not os.path.exists("results"):
        os.mkdir("results")

    best_acc = 0.0

    if cluster_vis:
        model.load_state_dict(
            torch.load(f"results/{save_name_pre}_model.pth", map_location="cpu")
        )
        tsne_cluster(
            model, test_loader, filename=f"results/{save_name_pre}_tsne_cluster_vis.png"
        )

    elif calc_euc_dist:
        model.load_state_dict(
            torch.load(f"results/{save_name_pre}_model.pth", map_location="cpu")
        )
        euc_dist_test(model, train_loader)

    else:
        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer)
            if epoch % 5 == 0:
                results["train_loss"].append(train_loss)
                test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
                results["test_acc@1"].append(test_acc_1)
                results["test_acc@5"].append(test_acc_5)

                # Save statistics.
                data_frame = pd.DataFrame(data=results, index=range(5, epoch + 1, 5))
                data_frame.to_csv(
                    f"results/{save_name_pre}_statistics.csv", index_label="epoch"
                )

                if test_acc_1 > best_acc:
                    best_acc = test_acc_1
                    torch.save(model.state_dict(), f"results/{save_name_pre}_model.pth")
            if epoch % 50 == 0:
                torch.save(
                    model.state_dict(),
                    f"results/{save_name_pre}_model_{epoch}.pth",
                )
