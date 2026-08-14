# Third-Party Code and Dataset Audit

Audit date: 2026-08-14

This document records the provenance and redistribution boundary for the Meta-Harness comparison.
It is an engineering compliance record, not legal advice. The machine-readable file identities are
in [`THIRD_PARTY.yml`](../../THIRD_PARTY.yml).

## Scope and release rule

The audit covers the external Meta-Harness text-classification runtime and the three datasets used
by the reported native comparison:

- Symptom2Disease (`symptom_diagnosis/`);
- the LawBench 3-3 charge-prediction subset (`crime_prediction/`); and
- the USPTO retrosynthesis subset (`uspto/`).

No item-level bytes from these datasets are tracked by SpiralHarness or included in its wheel.
Aggregate result summaries may be published, but prompts, targets, and item-level artifacts stay
outside Git. A dataset with an unknown license or an unresolved nested source must not be mirrored,
bundled, attached to a release, or uploaded as a SpiralHarness artifact. It may only be obtained on
demand from the pinned upstream for a use the operator is authorized to make.

This audit deliberately does not add a first-party root `LICENSE`; that choice must be made by the
SpiralHarness rights holder.

## First-party license blocker

The SpiralHarness repository has no root `LICENSE`, and its `pyproject.toml` has neither a project
`license` field nor `license-files` metadata. Third-party MIT or Apache-2.0 terms do not license the
first-party SpiralHarness implementation. Until the rights holder selects and adds a license, the
public repository is source-available under default copyright rather than an explicitly licensed
open-source release. This is a release and artifact-review blocker, but it is not authority for an
auditor to choose a license on the owner's behalf.

## Decision summary

| Component | Verifiable license evidence | SpiralHarness distribution decision |
|---|---|---|
| Meta-Harness reference runtime | MIT text at the pinned commit, copyright Yoonho Lee | Do not vendor; use a separate pinned checkout and first-party adapter |
| Symptom2Disease | Fixed Hugging Face dataset card declares Apache-2.0 | On-demand only in the current project; redistribution would require Apache-2.0 attribution and a modification notice |
| LawBench 3-3 subset | LawBench repository has Apache-2.0, but identifies CAIL2018 as the task's underlying source without pinned nested terms | **Do not redistribute** until the CAIL2018 data terms are verified |
| USPTO subset | No exact original release or dataset license is identified; the intermediate repository has no license text | **Do not redistribute**; license status is unknown |

The Meta-Harness repository's MIT license is valid evidence for its source code. It is not evidence
that Meta-Harness can relicense every third-party dataset it copied. The dataset analysis therefore
follows the provenance chain beyond Meta-Harness.

## Meta-Harness source runtime

The native comparison pins
[`stanford-iris-lab/meta-harness@44b9942`](https://github.com/stanford-iris-lab/meta-harness/tree/44b9942127847f7421db70d8c7e48407f09a3c70).
Its root
[`LICENSE`](https://raw.githubusercontent.com/stanford-iris-lab/meta-harness/44b9942127847f7421db70d8c7e48407f09a3c70/LICENSE)
contains the MIT license and `Copyright (c) 2026 Yoonho Lee`. The native experiment used these
upstream paths:

- `reference_examples/text_classification/benchmark.py`;
- `reference_examples/text_classification/inner_loop.py`;
- `reference_examples/text_classification/llm.py`;
- `reference_examples/text_classification/memory_system.py`;
- `reference_examples/text_classification/agents/no_memory.py`;
- `reference_examples/text_classification/agents/fewshot_all.py`;
- `reference_examples/text_classification/data/constants.py`;
- `reference_examples/text_classification/data/loaders.py`; and
- `reference_examples/text_classification/data/evaluators.py`.

Their byte hashes are recorded in `THIRD_PARTY.yml`. SpiralHarness does not contain copies of these
files. [`benchmarks/meta_harness_native/spiral_native.py`](../../benchmarks/meta_harness_native/spiral_native.py)
is a first-party bridge that is copied into a separately obtained upstream checkout at run time.

The external checkout used during the experiment was not pristine. Its `inner_loop.py` retained
item-level predictions in the external result object and passed `LITELLM_API_KEY` from the process
environment into the existing `LLM` constructor. The prediction retention also added fields to the
validation/test summaries. These local edits are not distributed by SpiralHarness. Any reproduction
must either reapply and disclose them or use a clean upstream runner and independently preserve the
same audit records. Candidate memory systems placed in the external checkout were SpiralHarness
experiment code, not unmodified upstream baselines.

### Official implementation versus paper reconstruction

The pinned public tree contains the official runtime, loaders, graders, `no_memory.py`,
`fewshot_all.py`, and `fewshot_memory.py`. It does not contain the paper-named discovered agents
`label_primed_query_anchored.py` or `draft_verification.py`. SpiralHarness's
[`meta_harness_paper.py`](../../spiral_harness/benchmark/meta_harness_paper.py) implements an
auditable reconstruction from Appendix B.1 and explicitly fills paper-unspecified choices; it is
not copied official source and is not a source-level reproduction.

Consequently, the native no-memory and few-shot comparisons may be described as comparisons with
official released baselines under the official runtime. Results against Label-Primed Query or Draft
Verification must be described only as comparisons with a **paper-algorithm reconstruction**. They
cannot support “SpiralHarness outperforms the official Meta-Harness system” unless the authors later
release the selected agent source or an independently runnable official artifact is evaluated under
the controlled protocol.

## Shared import chain

At the pinned Meta-Harness revision, the
[`data/README.md`](https://github.com/stanford-iris-lab/meta-harness/blob/44b9942127847f7421db70d8c7e48407f09a3c70/reference_examples/text_classification/data/README.md)
says the three directories were copied from `metaevo-ai/mce-artifact`. They entered Meta-Harness in
commit
[`95175f7`](https://github.com/stanford-iris-lab/meta-harness/commit/95175f70c758dd1145b395edfe8b67e6f9d80fbd).

All nine split files are byte-identical to the files introduced in
[`metaevo-ai/mce-artifact@0ff558d`](https://github.com/metaevo-ai/mce-artifact/tree/0ff558d0f64ffe61b9af6fc62ec3fa8b5a1440c1),
and those data paths remain unchanged at audit revision
[`c63300d`](https://github.com/metaevo-ai/mce-artifact/tree/c63300daef93404e22e2569b3d08e72843e7123b).
Neither revision has a root `LICENSE` or `NOTICE` file. At the data-introduction commit,
`pyproject.toml` contains `license = {text = "MIT"}` but the README does not contain a license grant;
the later README merely adds the words “MIT License.” Package metadata or a label without license
text and a clear licensor cannot establish rights in third-party dataset bytes. This is why the
audit continues to each original dataset source.

## Symptom2Disease

The intermediate
[`data/README.md`](https://github.com/metaevo-ai/mce-artifact/blob/0ff558d0f64ffe61b9af6fc62ec3fa8b5a1440c1/env/symptom_diagnosis/data/README.md)
identifies `gretelai/symptom_to_diagnosis` as its source. The fixed source revision is
[`722cfb0`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/tree/722cfb0e11f8ae37339c7f573b5e10429b94df49),
whose
[`README.md`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/blob/722cfb0e11f8ae37339c7f573b5e10429b94df49/README.md)
declares `license: apache-2.0` and says the dataset is licensed Apache-2.0. It also discloses that it
was adapted from a Kaggle Symptom2Disease dataset using an LLM.

The audit reproduced the provenance relationship rather than relying on matching names:

- all 212 Meta-Harness test records equal the fixed Hugging Face test split, in the same order,
  after renaming `input_text` to `question` and `output_text` to `answer`;
- all 250 Meta-Harness train/validation records are unique, exact renamed members of the 853-record
  Hugging Face train split; and
- the intermediate README describes a stratified 200/50 selection, although the referenced
  `prepare_data.py` is absent from the audited repository tree, so the selection procedure itself
  is not reproducible from that checkout.

Decision: the source license is verifiable. The project nevertheless keeps the data on demand. If
the transformed split is ever redistributed, include Apache-2.0 terms and attribution, mark the
field renaming and sampling as modifications, and re-check the upstream data card at the pinned
revision.

## LawBench charge prediction

The intermediate
[`crime_prediction/data/README.md`](https://github.com/metaevo-ai/mce-artifact/blob/0ff558d0f64ffe61b9af6fc62ec3fa8b5a1440c1/env/crime_prediction/data/README.md)
says its 350 records were selected from `3-3.json`. Exact-object comparison identifies that file as
LawBench
[`data/zero_shot/3-3.json`](https://github.com/open-compass/LawBench/blob/e30981bb3ff54c41571f222e0b23e92d27375388/data/zero_shot/3-3.json):

- the original contains 500 objects;
- the three Meta-Harness splits contain 350 unique objects; and
- every selected object is an exact member of the original file.

The source file was introduced in LawBench commit
[`8053ec8`](https://github.com/open-compass/LawBench/commit/8053ec8dc45ffb894d21dab34e9821f688e90359)
and is unchanged at audit revision
[`e30981b`](https://github.com/open-compass/LawBench/tree/e30981bb3ff54c41571f222e0b23e92d27375388).
LawBench has an
[`Apache-2.0 LICENSE`](https://raw.githubusercontent.com/open-compass/LawBench/e30981bb3ff54c41571f222e0b23e92d27375388/LICENSE).
However, its
[`README_EN.md`](https://github.com/open-compass/LawBench/blob/e30981bb3ff54c41571f222e0b23e92d27375388/README_EN.md)
identifies CAIL2018 as the data source for task 3-3 and does not pin a CAIL2018 revision or state the
nested dataset terms. The intermediate repository also omits the selection algorithm.

Decision: do not redistribute the 350-record split until the applicable CAIL2018 terms and the
right to republish this derived subset are confirmed. Reproduction may fetch the pinned
Meta-Harness copy on demand; the copy must remain outside source distributions, wheels, paper
supplements, and public run artifacts.

## USPTO retrosynthesis

The Meta-Harness split files are byte-identical to
`mce-artifact/env/uspto/data/{train,val,test}.jsonl` at commit `0ff558d`. The only upstream source
statement found is the task instruction's name “USPTO-50k.” There is no dataset README, original
repository URL, original revision, split-construction script, copyright notice, or dataset-specific
license. The intermediate repository's incomplete MIT metadata cannot establish the license of
USPTO-50k-derived records, and Meta-Harness's root MIT license cannot repair that missing chain.

Decision: license status is **unknown**. Do not redistribute, mirror, attach, or package any of the
three files. On-demand use is permitted by project policy only when the operator independently has
authority to use the upstream data. Before a public artifact release, obtain from the Meta-Harness
or MCE authors:

1. the exact original USPTO-50k repository and revision;
2. the script and seed that produced the 50/30/100 split;
3. the applicable dataset license or written redistribution permission; and
4. required attribution or notice text.

## Release checks

Before every release or paper-artifact upload:

1. verify no `symptom_diagnosis`, `crime_prediction`, or `uspto` JSONL bytes are tracked or included
   in the sdist/wheel;
2. keep item-level prompts, targets, predictions, and caches out of Git;
3. retain only aggregate results whose disclosure does not reconstruct protected dataset content;
4. re-run the SHA-256 checks in `THIRD_PARTY.yml` for any on-demand copy;
5. preserve Meta-Harness's MIT attribution if upstream source is redistributed separately;
6. preserve Apache-2.0 attribution and modification notices for any authorized Symptom2Disease
   export; and
7. keep LawBench and USPTO redistribution blocked until the open provenance items above are
   resolved and recorded in this file.
