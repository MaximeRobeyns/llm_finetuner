.. _usage:

..
    .. automodule:: llm_vae.mc_question_answering

Usage
=====

Basic Usage
-----------

The procedure for using this library involves:

1. Loading a pre-trained language model
2. deciding which modules in your model you'd like to quantize to 8 bit (note,
   only ``nn.Linear`` and ``nn.Embedding`` modules can be quantized at the
   moment.).
3. optionally, explicitly quantizing the pretrained model to save memory
4. creating one or more fine-tuned models

1. Loading a Pre-Trained Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This follows the standard procedure for loading models from e.g. HuggingFace.

.. code-block:: python

   import transformers

   model_name = 'facebook/opt-1.3b'
   base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

However, if you face memory issues, you can install `HF accelerate
<https://github.com/huggingface/accelerate>`_ (``pip install accelerate``) to
use ``low_cpu_mem_usage=True`` and also load the memory in ``float16``:

.. code-block:: python

   base_model = transformers.AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=t.float16,
       low_cpu_mem_usage=True,
       use_cache=False,
   )

Please don't load the model with HuggingFace's recent ``load_in_8bit=True`` as
this will interfere with ``finetuna``. Of course if you are only interested in
quantization, then you should just use this feature and not use ``finetuna``!

2. Viewing Adaptable Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``finetuna`` fine-tunes pre-trained models by freezing the pre-trained weights
(quantized or not) in each network module, and adding low-rank adapters on top
of these modules.

By default, the library will add adapters to *all* the layers, but this can be
unnecessary and you can save a lot of memory and computation by scrupulously
selecting the modules you add adapters to.

To get a list of your model's modules that you can add adapters to, first
quantize the model, then use the ``get_lora_adaptable_modules`` helper function:

.. code-block:: python

   import finetuna as ft

   ft.prepare_base_model(base_model)
   print(ft.get_lora_adaptable_modules(base_model))

3. Quantizing the Model
~~~~~~~~~~~~~~~~~~~~~~~

Quantizing the model refers to turning all (or a subset of) the frozen
pretrained model weights to int8 using the quantization scheme described in
`LLM.int8() <https://arxiv.org/pdf/2208.07339.pdf>`_.

This is an optional step and will be done automatically when creating new
finetuned models in the next step if ``base_model`` is not yet quantized.

If however you have memory constraints that mean that you can't keep a full
model loaded into memory, then use ``quantize_base_model(base_model)`` to
convert all the ``nn.Linear`` and ``nn.Embedding`` layers to 8bit:

.. code-block:: python

   ft.quantize_base_model(base_model)

.. note::

    The ``quantize_base_model`` function will modify the ``base_model``
    in-place, although it returns a reference to it for convenience.

This function also accepts an additional ``modules_not_to_freeze`` argument:
this does what it says on the tin, and doesn't quantize the modules listed in
this set. By default, this is set to ``lm_head`` (a module name shared by
``GPT-`` and ``OPT-`` models in HuggingFace), since we often want to retain full
precision for the language model head.

If you *do* want to quantize the language modelling head, you can set this to
teh empty set:

.. code-block:: python

   ft.quantize_base_model(base_model, modules_not_to_quantize=set())

Also note that if you quantize a module in the ``quantize_base_model`` function,
subsequently requiring that this module is no longer quantized when calling
``new_finetuned`` will result in an error. Later versions of ``finetuna``
may support this, but the loss of accuracy owing to the round-trip from
``float32`` -> ``int8`` -> ``float32`` is clearly sub-optimal. Un-quantized
modules *can* however later be quantized ad-hoc in ``new_finetuned``.

As a result, if you think you may require a module not to be quantized in the
future, it is safer to add it to the ``modules_not_to_quantize`` set, assuming
you have the memory overhead.

4. Creating Fine-Tuned models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have the base model in hand, we are ready to create some new models
to fine-tune using the ``new_finetuned`` function. The most basic invocation,
called without arguments will

- freeze and quantise all the pretrained ``nn.Linear`` and ``nn.Embedding``
  modules (with ``lm_head`` in the ``modules_not_to_quantize`` set by default).
  If you previously quantized the base model, then this step is skipped.
- add LoRA adapters to all Embedding and Linear layers using the default adapter
  configs (once again, the unquantized ``lm_head`` is treated as an exception by
  default, and optimised directly in its original datatype).
- all other ``base_model`` parameters which cannot be adapted are frozen

.. code-block:: python

   ft1 = ft.new_finetuned(base_model)

Using the ``adapt_layers`` argument
```````````````````````````````````

If you only wish to adapt certain layers, then you can specify these layers in
the ``adapt_layers`` argument:

.. code-block:: python

   ft1 = ft.new_finetuned(base_model, adapt_layers={"q_proj", "v_proj"})

In the above, we

- freeze and quantize all pretrained Embedding and Linear layers in
  ``base_model`` (excluding ``lm_head``)
- add LoRA adapters to ``q_proj`` and ``v_proj`` matrices only
- freeze everything else

In general, adapting just the query and value projection matrices in the
attention modules will be effective in fine-tuning the model, while greatly
decreasing the memory and computation required to do so.

See Section 7.1 of the `LoRa paper <https://arxiv.org/pdf/2106.09685.pdf>`_ for
a discussion of which layers are worth adapting.

Using the ``plain_layers`` argument
```````````````````````````````````

Occasionally, we want to keep a layer the same as in the base model, and
fine-tune it directly.

The running example of this has been the ``lm_head`` module, which is not frozen
nor quantized by default. When we call ``opt.step()``, we update its parameters
directly, not its adapter.

You can specify other layers to keep exactly as in the underlying ``base_model``
by adding them to the ``plain_layers`` argument when creating a new finetuned
model:

.. code-block:: python

   ft2 = ft.new_finetuned(
       base_model,
       adapt_layers={"q_proj", "v_proj"},
       plain_layers={"lm_head", "out_proj", "layer_norm"}
   )

In the above, we

- freeze and quantize all pretrained Embedding and Linear layers in
  ``base_model`` (excluding ``lm_head``, ``out_proj`` and ``layer_norm``)
- add LoRA adapters to ``q_proj`` and ``v_proj`` matrices only
- freeze all ``base_model`` parameters, except those in ``lm_head``,
  ``out_proj`` and ``layer_norm``.

Speicfying Adapter Configurations
``````````````````````````````````

By default, we use a LoRA adapter with a rank of 4 (``r=4``) and a scaling
factor of :math:`\alpha / r`, where ``alpha=1``. For Linear adapters, we
additionally set the dropout layer ``p=1``, and use a bias.

For Embedding adapters, the ``embedding_config`` argument to ``new_finetuned``
can either be:

- ``None``, in which case the following default configuration is used::

        EmbeddingAdapterConfig(r=4, alpha=1)

- A single ``EmbeddingAdapterConfig``, which is applied to all Embedding layers
  to adapt.
- A dictionary of type ``dict[str, EmbeddingAdapterConfig]``, which specifies
  the adapter configuration for *each* module to adapt. An error is raised if a
  module is left out.

Similarly, for Linear adapters, the ``linear_config`` argument can also be
``None``, a single ``LinearAdapterConfig``, or a dictionary of type ``dict[str,
LinearAdapterConfig]``.

See Section 7.2 of the `LoRa paper <https://arxiv.org/pdf/2106.09685.pdf>`_ for
a discussion of what rank to use. In summary, performance seems to improve
through ``r=1``, ``r=2`` and plateaus at ``r=4`` before falling back down at
``r=8``. Setting the rank to be very high like ``r=64`` yields no benefit.

Owing to the size of the layers, a lot of memory and computation can be saved
for each incremental decrease in ``r``.

More Controls
`````````````

..
    Layers can either be:
    - quantized to 8-bits or not
    and
    - frozen, not-frozen or adapted

    This gives 6 combinations. By default, we want to quantize everything we
    can, and add adapters to everything. However, when deviating from this
    default, there are two main types of layers we care about when fine-tuning:

    1. The first are layers we want to add LoRA adapters to, such as
    query and value projection matrices in the attention module. That is, the
    pretrained weights are quantized and frozen, and we train low-rank, FP32
    'adapters'. You can choose which layers we treat like this by naming them
    in the ``adapt_layers`` argument,

    2. The second are layers that we just want to leave alone, such as the
    language model head. By 'leave alone', we mean that these are vanilla NN
    layers: not quantized and not-frozen (nor adapted). You can list the layers
    to treat as such in the ``plain_layers`` argument. These cannot intersect
    with ``adapt_layers``.

    To give full control over what happens to each layer, we also provide the
    modules_not_to_freeze and modules_not_to_quantize arguments.


The options described thus far shuold be all you need for most cases. The
options in this section should not have to be used very often.

The ``new_finetuned`` function also has two other arguments called
``modules_not_to_freeze`` and ``modules_not_to_quantize``.

The ``plain_layers`` argument is really just for convenience, and inserts its
contents into both ``modules_not_to_freeze`` and ``modules_not_to_quantize``.
Using these two arguments however allows you more fine-grained control, such as
adapting a non-quantized layer.

Note that for now it is an error to:

1. place a module in ``modules_not_to_quantize`` if it has previously been
   quantized in the base model during a call to ``quantize_base_model``.
2. place a module in ``modules_not_to_freeze`` if it is quantized (directly
   optimising int8 weights is possible, and will be supported in the future)


For completeness, the full signature of the ``new_finetuned`` function is:

.. autofunction:: finetuna.new_finetuned
