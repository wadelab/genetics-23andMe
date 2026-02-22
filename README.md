# genetics-23andMe
Python tools to parse and analyze raw 23andMe data, including a GUI for SNP review
and Found My Fitness matches.

## Setup
This project uses `uv` for the virtual environment and dependencies.

   uv sync

This creates a local `.venv` and installs all dependencies.

## CLI Usage
From a directory containing your raw 23andMe data file(s):

   uv run python -m genetics_23andme.core

Raw data should be in a txt file starting with `genome_` and ending with `.txt`.
Example: `genome_Example_v4_Full_20190408081640.txt`

The program will scan the current directory, show matching files, and prompt you
to select one for processing. It will also compare against Found My Fitness
noteworthy SNPs and report matches.

## GUI Usage
To run the interactive GUI for browsing SNP summaries and Found My Fitness matches:

   uv run python -m genetics_23andme.gui

## Optional: SNPedia + OpenAI Synthesis
The GUI can fetch SNPedia summaries. If you want AI synthesis of the results,
set the `OPENAI_API_KEY` environment variable before launching the GUI.

## Project Layout
```
.
├── pyproject.toml
├── uv.lock
├── AGENTS.md
├── README.md
│
├── src/
│   └── genetics_23andme/
│       ├── __init__.py
│       ├── core.py
│       ├── gui.py
│       └── fitness.py
│
├── tests/
│   └── test_core.py
│
└── docs/
   ├── plan.md
   └── adr/
```

## Legacy Files
The original scripts were moved to [legacy/](legacy/) for reference:
- `genome.py`
- `genome_gui.py`
- `foundMyFitness.py`

New development should use the `src/genetics_23andme/` modules.

## Sample Output

   Opening: genome_Example_v4_Full_20190408081640.txt

    Processing 601886 SNPs sorted by Chromosome
     1:..............................................
     2:..............................................
     3:.......................................
     4:..................................
     5:..................................
     6:........................................
     7:..................................
     8:..............................
     9:..........................
    10:..............................
    11:.............................
    12:............................
    13:......................
    14:...................
    15:..................
    16:...................
    17:...................
    18:................
    19:.............
    20:...............
    21:........
    22:..........
    X?:.....................
    MT:........

    Detected: 24 chromosomes, 601886 positions, 20 genotypes

    rsID count: 553292, Internal ID count: 43511

    Calculating number of unique occurrences per genotype

    Genotype: CC = 109444
    Genotype: GG = 109046
    Genotype: AA =  88377
    Genotype: TT =  87813
    Genotype: CT =  66297
    Genotype: AG =  65760
    Genotype: -- =  16904
    Genotype: AC =  15145
    Genotype: GT =  14993
    Genotype: G  =   4944
    Genotype: C  =   4812
    Genotype: A  =   4198
    Genotype: T  =   3911
    Genotype: II =   2369
    Genotype: DD =    810
    Genotype: CG =    803
    Genotype: I  =    528
    Genotype: AT =    434
    Genotype: D  =    172
    Genotype: DI =     43

    Number of positions per chromosome pair

    Chromosome: 1  = 46661
    Chromosome: 2  = 46128
    Chromosome: 6  = 40384
    Chromosome: 3  = 38517
    Chromosome: 5  = 34384
    Chromosome: 4  = 33915
    Chromosome: 7  = 33053
    Chromosome: 8  = 30268
    Chromosome: 11 = 29323
    Chromosome: 10 = 29211
    Chromosome: 12 = 28451
    Chromosome: 9  = 26586
    Chromosome: X? = 21780
    Chromosome: 13 = 21679
    Chromosome: 16 = 19200
    Chromosome: 17 = 18743
    Chromosome: 14 = 18695
    Chromosome: 15 = 18281
    Chromosome: 18 = 16490
    Chromosome: 20 = 14494
    Chromosome: 19 = 13001
    Chromosome: 22 =  9098
    Chromosome: 21 =  8461
    Chromosome: MT =  5083

    Scanning Found My Fitness data for possible issues

    MT rs1801131  GT:  AA  MTHFR      Risk for altered folate metabolism and hyperhomocysteinemia
       See: https://www.snpedia.com/index.php/rs1801131

    MT rs1061170  CT:  CT  CFH        Normal lifespan; slightly increased risk for macular degeneration
       See: https://www.snpedia.com/index.php/rs1061170

    MT rs7571842  AA:  AA  SLC4A5     Increased risk for salt sensitivity of blood pressure
       See: https://www.snpedia.com/index.php/rs7571842

    MT rs2305160  GG:  CC  NPAS2      Circadian-associated increased breast/prostate cancer risk
       See: https://www.snpedia.com/index.php/rs2305160

    ...
