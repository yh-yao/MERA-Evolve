# Skill: humaneval

## When to use
Task is a HumanEval-style problem: a complete function signature and docstring, often with worked examples written as doctests.

## Procedure
1. Read the docstring's examples carefully -- they define the exact expected behavior, including edge cases.
2. Keep the given function name and signature exactly as written.
3. Handle boundary cases explicitly: empty input, negative numbers, single-element input, boundary values.
4. Return one self-contained function; do not add extra top-level code, explanations, or tests.
5. Prefer a direct, correct implementation over a clever one-liner.

## Common pitfalls
- When the output must preserve original separators or spacing, don’t use plain `split()`/`join()` unless collapsing whitespace is allowed; tokenize while keeping delimiters.
- For rank selection among distinct values, deduplicate before indexing sorted results; but for filtered list outputs, don’t deduplicate just because the function name says “unique” unless the spec requires it.
- For filtered extrema, guard empty inputs and empty filtered candidate sets before calling `min`, `max`, indexing sorted results, or returning paired metadata; use the required sentinel.
- For iterative arithmetic checks, handle degenerate cases up front: base `0`/`1`, nonpositive targets, reversed ranges, and minimum feasible bounds can otherwise cause infinite loops or false positives.

## Recurring patterns
- Use helper predicates for reusable checks such as primality, palindrome status, digit constraints, sign class, integer-ness, or length bounds, then filter/count with comprehensions.
- For closest pairs, rank selection, and order-dependent comparisons, sort first and inspect adjacent elements or indexed positions instead of comparing every pair.
- Encode multi-criteria ordering with tuple keys, e.g. `(primary_measure, secondary_value)`, and rely on strict updates when “first occurrence” or “smallest index” should win ties.
- For fixed character classes or symbolic categories, use sets/maps directly: prime hex digits, vowels, odd digits, sign counts, roman numeral values, or known ordered names.

---

# Skill: mbpp

## When to use
Task is an MBPP-style problem: the goal is a plain-English sentence, and the exact function name/signature is only implied by the assert statements that follow the prompt.

## Procedure
1. Infer the exact function name and argument order from the assert statements before writing any code.
2. Solutions are typically short, direct algorithms (loops, basic string/list operations); avoid over-engineering or unnecessary abstractions.

## Common pitfalls
- Functions declared with `*args` often still have a fixed expected argument layout; unpack operands by position first, especially when extra length/index parameters are included.
- Using `set` for union or symmetric difference removes duplicates but also destroys sequence order; if the expected result is tuple/list-like, preserve first-seen order explicitly.
- List multiplication with mutable elements aliases the same object repeatedly, e.g. `[{}] * n`; use a comprehension to create independent containers.
- For regex matches that need positions, use match objects from `finditer`; `str.index(match_text)` returns the first occurrence and breaks on repeated matches.

## Recurring patterns
- For simple extraction, filtering, conversion, flattening, or per-item transformation, use direct constructors/comprehensions: column projection, odd filtering, tuple conversion, character splitting, sorted sublists, flatten-then-join.
- For frequency-based questions, build a `Counter`, then derive the answer from counts: max-minus-min frequency, items appearing once, duplicate-aware filtering, or pair counts.
- For sorted-array selection or closest-combination tasks, use coordinated pointers/merge logic and advance the pointer holding the limiting value instead of brute-forcing all combinations.
- For subsequence or contiguous-subarray length/sum problems, keep compact DP state: per-index LIS/LDS arrays for subsequences, or rolling “best ending here” state for contiguous sums.
