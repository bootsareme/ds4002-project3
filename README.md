# ds4002-project3
AI-generated meme slop

## Section I: Software and platform selection
For this project, we used a Unix-like OS (Linux/Mac OS). Python 3 must be installed and the following dependencies as well:
```
pip install torch torchvision transformers pandas pillow openpyxl
```

## Section 2: A Map of our documentation
```
.
├── DATA
│   ├── Brain.png
│   ├── Buttons.png
│   ├── Exit swerving.png
│   ├── Fence.png
│   ├── Gru.png
│   ├── Spiderman.png
│   ├── Wolf.png
│   └── meme compilation.xlsx
├── LICENSE
├── OUTPUT
│   ├── all_similarity_scores.csv
│   ├── avg_comparison.png
│   ├── avg_scores_blip.png
│   ├── avg_scores_blipv2.png
│   ├── query_similarity_summary.csv
│   ├── results_blip.csv
│   ├── results_blipv2.csv
│   └── total_avg_comparison.png
├── README.md
└── SCRIPTS
    ├── BLIP_caption_gen.py
    └── BLIPv2_caption_gen.py
    └── SBERT_analysis.py
```

## Section 3: Instructions
1. Go to Rivanna's OOD: ood.hpc.virginia.edu
2. Reserve an 8-core, 16 GB instance and login.
3. In the terminal, activate a virtual env: `python3 -m venv venv`
4. `source venv/bin/activate`
5. Run `python3 SCRIPTS/BLIPv2_caption_gen.py` or the other scripts in similar fashion.
6. The output will be saved in the appropriatly titled directory (it may take a while).
