# single_file_gpt
A character-level GPT from scratch in a single file [gpt.py](https://github.com/veezbo/single_file_gpt/blob/main/gpt.py). 

Optimized for readability and learnability.

## features
- single file
- as readable as possible
- comments for learnings and common errors
- [type annotations](https://github.com/patrick-kidger/jaxtyping)
- working code that trains on text and generates text like it

## demo
We train a character-level GPT on a small corpus of Shakespearian English from plays.

After training, the same model is used to generate similar text, especially reproducing the style and syntax of the input.

## dependencies
```
python >= 3.10
```
```
torch >= 2.0
jaxtyping==0.2.28
```

## install

```
uv sync
```

## run
```
uv run python gpt.py
```

Running `gpt.py` trains the model, prints periodic train/validation loss estimates, and then generates text from the trained model. Runtime depends heavily on whether PyTorch is using a GPU or CPU.

The module can also be imported without starting training:
```python
from gpt import GPTLanguageModel
```

## typing
This project uses [jaxtyping](https://github.com/patrick-kidger/jaxtyping) for type annotations.

This is what the type annotations look like:
```python
from torch import Tensor
from jaxtyping import Float, Int

def func(x: Float[Tensor, "A B C"]) -> Int[Tensor, ""]:
    return x.shape[0]

Float[Tensor, "A B C"]  # float tensor with shape (A, B, C)
Int[Tensor, ""]  # int scalar (0-dim) tensor
func  # function that takes in a float tensor with shape (A, B, C) and returns an int scalar tensor
```


## contributing
All contributions in the form of confusions, concerns, suggestions, or improvements are welcome!

## future
- include type annotations for all variables within functions too when this is well-supported by jaxtyping, see this [issue](https://github.com/patrick-kidger/jaxtyping/issues/153)

## acknowledgements
This repo is heavily influenced by Andrej Karpathy's [nanogpt](https://github.com/karpathy/nanoGPT/tree/master)

## license
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
