Please download the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset.

Run the following script to convert the fine-grained facial component masks provided by CelebAMask-HQ into the binary masks used in our paper:

```bash
python data/process_dataset.py
```

The generated binary masks will be saved in the `mask_images/` directory.

