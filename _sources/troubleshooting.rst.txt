.. _troubleshooting:

Troubleshooting Your Installation
=================================

Here are some issues folks have run into and what they did to resolve them.


PyTorch & CUDA Woes
-------------------

You need PyTorch installed with ``cudatoolkit`` (see
`link
<https://docs.google.com/document/d/1JxSo4lQgMDBdnd19VBEoaG-mMfQupQ3XvOrgmRAVtpU/edit#>`_).

You don't seem to get ``cudatoolkit`` when you install PyTorch using ``pip``.
You may be able to install ``cudatoolkit`` separately, but we haven't tested
that. Our tested proceedure is to remove previous installs of PyTorch.  e.g. if
PyTorch is currently installed through pip:

.. code-block:: bash

    pip uninstall torch

Then, install PyTorch with ``cudatoolkit`` through conda (see
`the PyTorch installation page <https://pytorch.org/get-started/locally/>`_ for
the latest install command, but CUDA 11.3 seems to be working with
``bitsandbytes``)

.. code-block:: bash

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

Then clone this repo, navigate to the repository root and use:

.. code-block:: bash

    pip install -e .

Notes on bitsandbytes
---------------------

``bitsandbytes`` was forked in a confusing way.

The actual repo is https://github.com/TimDettmers/bitsandbytes, and this version
installs with

.. code-block:: bash

    pip install bitsandbytes

However, if you search on Google, you get the older repo
https://github.com/facebookresearch/bitsandbytes and this version installs with

.. code-block:: bash

    pip install bitsandbytes-cudaXXX

We need the newer version, installed with ``pip install bitsandbytes``.

Some Handy Dandy Environment Variables
--------------------------------------

Stick these in your shell's config for prosperity:


.. code-block:: bash

    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    export CUDA_VISIBLE_DEVICES=0

The first will prevent the GPU memory from getting too fragmented.
The memory allocator was probably not tuned for working with the huge LLMs *du
jour* and tends to get upset when we load massive models.

The second line is not entirely necessary but can be useful when you aren't
going to use all the GPUs on your machine / node. This is helpful if you want to
run several experiments in parallel on different processes. (The example only
makes CUDA:0 available, but you can of course list more GPUs here.)
