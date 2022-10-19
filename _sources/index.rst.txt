``finetuna`` Documentation
==========================

**Fine-tune multiple LLMs in resource-constrained environments.**

Introduction
------------

This is a library to fine-tune large langague models in resource-constrained
environments.

This works by freezing the large pretrained model weights, and quantizing them
to 8 bits to save memory. Low-rank adapters are then added to certain network
modules and only these are optimised.

Once fine-tuned, you can export this quantized and adapted model to the same
format as the original pretrained model weights (in either ``float16`` or
``float32``).

The approach taken allows you to hold multiple fine-tuned LLMs in memory using
only marginally more memory than it would cost to hold a single pretrained model
in memory. This is useful when writing probabilistic programs or 'cascades' with
language models.

The two optimisations used are:

- `8-bit quantization <https://arxiv.org/pdf/2110.02861.pdf>`_ of pre-trained model weights, and
- fine-tuning with `LoRA adapters <https://arxiv.org/pdf/2106.09685.pdf>`_.

You can find a high-level description of the optimisations on the `overview page
<overview_of_optimisations.html>`_.


Installation
------------

We recommend that you work within a virtual environment (e.g. conda or venv).

**Prerequisite**: PyTorch with CUDA support (11.3 recommended, but will work
with other versions)

Clone the repository, navigate to root, and do

.. code-block:: bash

    make install

If you run into issues, see our `troubleshooting <troubleshooting.html>`_ guide.

For development, to install dependencies for

- tests, run ``pip install -e .[test]``
- documentation, run ``pip install -e .[docs]``
- everything, run ``pip install -e .[all]``.

Basic Usage
-----------

At the simplest level, you initialise a HuggingFace transformer model (or any
other ``nn.Module``), and wrap it with ``finetuna.new_finetuned(my_model)``.

.. code-block:: python

   import torch as t
   import finetuna as ft
   import bitsandbytes as bnb
   import transformers

   # Setup a base model
   model_name = "EleutherAI/gpt-j-6B"
   base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

   # Wrap it (you can call new_finetuned on the same base_model more than once)
   finetuned = ft.new_finetuned(base_model)

   # Setup an optimiser *from the bitsandbytes library*
   opt = bnb.optim.AdamW(finetuned.parameters())

   # Run your usual training loop with an automatic mixed precision context:
   with t.cuda.amp.autocast():
        opt.zero_grad()
        loss = mse_loss(finetuned(x) - y)
        finetuned.backward(loss)
        opt.step()

Note that the above is very simplistic and, while illustrative, will probably
not give you very good results.

Please read on to the `usage guide <usage.html>`_ for a more comprehensive
guide.

Also be sure to stop by the `overview page <overview_of_optimisations.html>`_ to
see how the optimisations work.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   overview_of_optimisations
   troubleshooting

..
   fine_tuning
   question_answering

..
    Indices and tables
    ------------------

    * :ref:`genindex`
    * :ref:`modindex`
