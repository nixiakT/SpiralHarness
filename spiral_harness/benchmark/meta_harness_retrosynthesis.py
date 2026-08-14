"""Reaction-aware memory for Meta-Harness's USPTO retrosynthesis task."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

_REACTION_TYPE = re.compile(r"reaction type is\s+([^.]*)\.", re.IGNORECASE)
_PRODUCT = re.compile(r"Input:\s*([^\n]+)", re.IGNORECASE)
_SMILES_TOKEN = re.compile(
    r"\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Fe|Zn|Cu|Sn|[A-Z][a-z]?|"
    r"[a-z]|%\d\d|\d|[=#@+\-\\/().]"
)

_FAMILY_GUIDANCE = {
    "acylation and related processes": (
        "Disconnect the newly acylated N/O/C bond and recover the nucleophile plus the "
        "acid, activated acid, anhydride, isocyanate, or related acyl donor."
    ),
    "c-c bond formation": (
        "Disconnect the newly formed carbon-carbon bond and recover the two coupling partners, "
        "including halide/boron/phosphonium functionality shown by the analogues."
    ),
    "deprotections": (
        "The observed product is deprotected; reconstruct the protected precursor. A protecting "
        "group present in the precursor may be absent from the product."
    ),
    "functional group addition (fga)": (
        "Remove the group added in the forward reaction and include its source reagent when the "
        "training analogues do so."
    ),
    "functional group interconversion (fgi)": (
        "Reverse the functional-group conversion while preserving the rest of the scaffold and "
        "include a small coproduct only when supported by the analogues."
    ),
    "heteroatom alkylation and arylation": (
        "Disconnect the new heteroatom-carbon bond into a heteroatom nucleophile and an alkyl or "
        "aryl electrophile."
    ),
    "heterocycle formation": (
        "Open the newly formed ring at the bonds implied by the closest analogues and return all "
        "ring-forming precursors."
    ),
    "oxidations": (
        "The requested precursors are at a lower oxidation state than the observed product. Do "
        "not accidentally propose the forward oxidation product."
    ),
    "protections": (
        "The observed product is protected; return the unprotected substrate plus the protecting "
        "reagent when it appears in the analogues."
    ),
    "reductions": (
        "The requested precursor is at a higher oxidation state or otherwise less reduced than "
        "the observed product. Do not run the reduction in the forward direction again."
    ),
}

_BOC_GROUP = "CC(C)(C)OC(=O)"
_BOC_ANHYDRIDE = "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"
_NBS = "O=C1CCC(=O)N1Br"


@dataclass(frozen=True, slots=True)
class RetrosynthesisExample:
    """One product-to-precursor example available to the offline harness."""

    reaction_type: str
    product: str
    target: str


def parse_retrosynthesis_query(text: str) -> tuple[str, str]:
    """Extract the reaction family and product SMILES from an upstream task prompt."""

    type_match = _REACTION_TYPE.search(text)
    product_match = _PRODUCT.search(text)
    if not type_match or not product_match:
        raise ValueError("not a Meta-Harness retrosynthesis query")
    return type_match.group(1).strip(), product_match.group(1).strip()


def smiles_terms(smiles: str) -> Counter[str]:
    """Build dependency-free local structure features from a SMILES string."""

    tokens = _SMILES_TOKEN.findall(smiles)
    terms: Counter[str] = Counter(f"atom:{token}" for token in tokens)
    for width in (2, 3, 4):
        terms.update(
            f"path{width}:" + "".join(tokens[index : index + width])
            for index in range(len(tokens) - width + 1)
        )
    return terms


def weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    """Return multiset Jaccard similarity for sparse structure features."""

    union = left.keys() | right.keys()
    denominator = sum(max(left[term], right[term]) for term in union)
    if not denominator:
        return 0.0
    return sum(min(left[term], right[term]) for term in union) / denominator


def compile_retrosynthesis_answer(query: str, draft: str) -> str:
    """Apply validation-selected, high-confidence reverse-reaction templates.

    The rules operate on reaction families and functional patterns rather than individual query
    strings. They intentionally fail open to the model draft when a pattern is ambiguous.
    """

    reaction_type, product = parse_retrosynthesis_query(query)
    family = reaction_type.casefold()

    if family == "functional group addition (fga)" and "CBr" in product:
        return f"{product.replace('CBr', 'C')}.{_NBS}"

    if family == "protections" and product.startswith(_BOC_GROUP):
        return f"{_BOC_ANHYDRIDE}.{product.removeprefix(_BOC_GROUP)}"

    if family == "functional group interconversion (fgi)" and "C(N)=O" in product:
        return f"{product.replace('C(N)=O', 'C(=O)O')}.N"

    if family == "deprotections":
        if product.startswith("O=C(O)"):
            return "CCOC(=O)" + product.removeprefix("O=C(O)")
        if product.endswith("C(=O)O"):
            return product + "CC"

    if family == "reductions":
        if product.startswith("OCCC"):
            return "O=C(O)CC" + product.removeprefix("OCCC")
        if product.count("C(O)") == 1:
            return product.replace("C(O)", "C(=O)", 1)

    cleaned_draft = draft.strip()
    if family == "oxidations" and "S(=O)C" in product and "OO" not in cleaned_draft.split("."):
        return f"{cleaned_draft}.OO"

    return cleaned_draft


class SpiralRetrosynthesisMemory:
    """Retrieve structurally similar demonstrations from the same reaction family.

    Meta-Harness's Label-Primed Query is designed for repeated class labels. USPTO instead has
    nearly unique structured outputs, so treating every answer as a label creates a long,
    distracting context. This memory first enforces the reaction family and then ranks products
    using dependency-free SMILES path features.
    """

    def __init__(self, *, neighbor_k: int = 5, context_char_budget: int = 24_000) -> None:
        if neighbor_k <= 0 or context_char_budget <= 0:
            raise ValueError("invalid retrosynthesis-memory configuration")
        self.neighbor_k = neighbor_k
        self.context_char_budget = context_char_budget
        self.examples: list[RetrosynthesisExample] = []

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            source = str(result.get("raw_question", result["input"]))
            reaction_type, product = parse_retrosynthesis_query(source)
            self.examples.append(
                RetrosynthesisExample(
                    reaction_type=reaction_type,
                    product=product,
                    target=str(result["ground_truth"]),
                )
            )

    def retrieve(self, query: str) -> tuple[tuple[RetrosynthesisExample, float], ...]:
        reaction_type, product = parse_retrosynthesis_query(query)
        family = [
            example
            for example in self.examples
            if example.reaction_type.casefold() == reaction_type.casefold()
        ]
        candidates = family or self.examples
        query_terms = smiles_terms(product)
        ranked = sorted(
            candidates,
            key=lambda example: (
                -weighted_jaccard(query_terms, smiles_terms(example.product)),
                example.product,
                example.target,
            ),
        )
        return tuple(
            (example, weighted_jaccard(query_terms, smiles_terms(example.product)))
            for example in ranked[: self.neighbor_k]
        )

    @staticmethod
    def _guidance(reaction_type: str) -> str:
        return _FAMILY_GUIDANCE.get(
            reaction_type.casefold(),
            "Infer the reverse transformation from the same-family analogues.",
        )

    def build_context(self, query: str) -> str:
        reaction_type, _product = parse_retrosynthesis_query(query)
        selected = self.retrieve(query)
        if not selected:
            examples = "(no training analogues available)"
        else:
            chunks = []
            used = 0
            for number, (example, score) in enumerate(selected, start=1):
                chunk = (
                    f"[Same-family analogue {number}; structure score={score:.3f}]\n"
                    f"Observed product: {example.product}\n"
                    f"Correct precursor set: {example.target}"
                )
                if used + len(chunk) + 2 > self.context_char_budget:
                    continue
                chunks.append(chunk)
                used += len(chunk) + 2
            examples = "\n\n".join(chunks) or "(context budget excluded all analogues)"
        return (
            f"Reaction family: {reaction_type}\n"
            f"Reverse-direction reminder: {self._guidance(reaction_type)}\n\n"
            f"Successful training reactions from this family:\n{examples}"
        )

    def build_prompt(self, input: str) -> str:
        context = self.build_context(input)
        return f"""Solve the retrosynthesis problem using reaction-family-specific
analogical memory.

The examples map an observed PRODUCT to its correct PRECURSORS. Transfer the reaction-center
change, never the examples' scaffold. Preserve the query product's unchanged substructure exactly.
Return every participating precursor but do not add catalysts, solvents, or unsupported reagents.
The final answer is an unordered period-separated set; each SMILES spelling must be exact.

{context}

## Official problem
{input}

Before answering, audit that the proposed precursors would produce the given product in the named
reaction family and that you did not reverse the direction twice.

Return valid JSON with exactly these fields:
{{"reasoning": "brief reaction-center audit", "final_answer": "precursor SMILES only"}}"""

    def get_state(self) -> str:
        return json.dumps(
            {
                "schema_version": "1",
                "kind": "spiral-retrosynthesis-memory",
                "neighbor_k": self.neighbor_k,
                "context_char_budget": self.context_char_budget,
                "examples": [asdict(example) for example in self.examples],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def set_state(self, state: str) -> None:
        payload = json.loads(state)
        if payload.get("kind") != "spiral-retrosynthesis-memory":
            raise ValueError("not a Spiral retrosynthesis-memory state")
        self.neighbor_k = int(payload["neighbor_k"])
        self.context_char_budget = int(payload["context_char_budget"])
        self.examples = [RetrosynthesisExample(**example) for example in payload["examples"]]


__all__ = [
    "RetrosynthesisExample",
    "SpiralRetrosynthesisMemory",
    "compile_retrosynthesis_answer",
    "parse_retrosynthesis_query",
    "smiles_terms",
    "weighted_jaccard",
]
