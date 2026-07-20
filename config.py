import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config():
    parser = argparse.ArgumentParser(
        description="Training and testing configuration for ID-Guard"
    )

    # ==========================================================
    # Runtime
    # ==========================================================
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "test_one"],
        help="Running mode.",
    )

    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )

    # ==========================================================
    # Target Deepfake Models
    # ==========================================================
    parser.add_argument(
        "--target_model",
        nargs="+",
        default=["stargan", "aggan", "fpgan", "relgan", "HiSD"],
        choices=["stargan", "aggan", "fpgan", "relgan", "HiSD"],
        help="Target deepfake models.",
    )


    # ==========================================================
    # Dataset
    # ==========================================================
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Input image size.",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="dataset/CelebAMask-HQ/CelebA-HQ-img/",
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        default="dataset/CelebAMask-HQ/mask_images/",
    )

    parser.add_argument(
        "--attr_path",
        type=str,
        default="dataset/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt",
    )

    # ==========================================================
    # Training
    # ==========================================================
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Mini-batch size.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Beta1 for Adam.",
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adam.",
    )

    parser.add_argument(
        "--dws",
        type=str,
        default="MGDA",
        choices=["MGDA", "KPI"],
        help="Dynamic weight strategy.",
    )

    # ==========================================================
    # Loss
    # ==========================================================
    parser.add_argument(
        '--id_extractor', 
        type=str, 
        default='Arc', 
        choices=["Vgg", "Arc"],
        help='the ID extractor model, in Vgg|Arc'
    )

    parser.add_argument(
        "--lambda_mask",
        type=float,
        default=1.0,
        help="Weight of mask loss.",
    )

    parser.add_argument(
        "--lambda_id",
        type=float,
        default=0.1,
        help="Weight of identity loss.",
    )

    parser.add_argument(
        "--lambda_ft",
        type=float,
        default=0.01,
        help="Weight of feature loss.",
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="both",
        choices=["bin", "hmp", "both"],
        help="Mask supervision type.",
    )

    parser.add_argument(
        '--mask_comb_type', 
        type=str, 
        default='multi', 
        choices=["add", "multi"],
        help='Combination types of two masks'
    )

    # ==========================================================
    # proactive defense
    # ==========================================================
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Perturbation budget.",
    )

    # ==========================================================
    # Config
    # ==========================================================
    parser.add_argument(
        "--deepfake_model_path",
        type=str,
        default="configs/deepfake_models.json",
    )

    parser.add_argument(
        "--prior_weight_path",
        type=str,
        default="configs/prior_weight.json",
    )

    parser.add_argument(
        "--id_model_path",
        type=str,
        default="configs/id_model.json",
    )

    # ==========================================================
    # Output
    # ==========================================================
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="checkpoint/ID-guard/",
    )

    parser.add_argument(
        "--sample_dir",
        type=str,
        default="samples/",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
    )

    parser.add_argument(
        "--test_image_path",
        type=str,
        default=None,
        help="Path to a single test image.",
    )

    parser.add_argument(
        "--save_iter",
        type=int,
        default=200,
        help="sample saving interval.",
    )


    return parser.parse_args()