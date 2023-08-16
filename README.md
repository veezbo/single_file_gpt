# single_file_gpt
A character-level GPT from scratch in a single file [gpt.py](https://github.com/veezbo/single_file_gpt/blob/main/gpt.py). 

Optimized for readability and learnability.

## features
- single file
- as readable as possible
- comments for learnings and common errors
- working code that trains on text and generates text like it

## demo
We train a character-level GPT on a small corpus of Shakespearian English from plays.

After training, the same model is used to generate similar text, especially reproducing the style and syntax of the input.

## dependencies
```
python >= 3.10
torch >= 2.0
```

## install

```
pip install torch
```

## run
```
python gpt.py
```

## contributing
All contributions in the form of confusions, concerns, suggestions, or improvements are welcome!

## acknowledgements
This repo is heavily influenced by Andrej Karpathy's [nanogpt](https://github.com/karpathy/nanoGPT/tree/master)

## license
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
