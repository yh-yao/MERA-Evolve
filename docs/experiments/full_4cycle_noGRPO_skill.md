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
- Guard before indexing, reductions, hashing, or parsing: empty strings/lists and short lists may require `False`, `None`, `[]`, `()`, or `0` instead of `s[-1]`, `array[0]`, `min()`, or conversions throwing.
- Decide whether duplicates are meaningful before using `set()` or removal: “second smallest” may require unique values, while alternating min/max or filtered numeric lists may need duplicate counts preserved.
- Resolve tie rules before selecting: equal best values may require keeping the first index, equal aggregate scores may return the first input, and equal maximum metrics may need lexicographic tie-breaking.
- Apply type, sign, range, and format constraints before digit/string logic: reject non-integers first, check thresholds like “greater than 10” before inspecting digits, and validate fraction/date/name formats before converting.

## Recurring patterns
- Use “filter candidates, then aggregate/select” with explicit empty handling: count matching chars/items, sum qualifying prefixes, or choose min/max among positives, negatives, primes, or even values.
- Use sort keys for derived ordering: sort by `(computed_metric, original_value)`, choose ascending/descending from endpoint parity, or sort once before extracting ordered extrema/proximity.
- Encapsulate reusable predicates/conversions in small helpers: primality, digit parity, uppercase/vowel tests, palindrome checks, Roman/hex mappings, and fraction/whole-number checks become composable.
- Use compact structural transforms when the task is textual: comprehensions for deletion/filtering, slicing for palindrome reversal, regex or run scanning for space groups, and doubled-string rotation checks for cyclic substrings.

---

# Skill: mbpp

## When to use
Task is an MBPP-style problem: the goal is a plain-English sentence, and the exact function name/signature is only implied by the assert statements that follow the prompt.

## Procedure
1. Infer the exact function name and argument order from the assert statements before writing any code.
2. Solutions are typically short, direct algorithms (loops, basic string/list operations); avoid over-engineering or unnecessary abstractions.

## Common pitfalls
- When a stub uses `*args`, examples still imply fixed roles and positions; unpack by meaning, such as `(sequence, index)`, `(array, n)`, or `(a, b)`, rather than iterating over all arguments as data.
- Using `set` union or symmetric difference may lose required first-seen ordering and duplicate-sensitive behavior; if the expected output is tuple/list-like, build it by scanning inputs in order.
- Avoid list multiplication for nested mutable objects like dictionaries or lists; `[{}] * n` creates repeated references, so use a comprehension when later mutation is possible.
- For regex/string position tasks, locating matched text with `str.index()` can return an earlier duplicate; use `finditer()` match spans for current-match positions.

## Recurring patterns
- For digit-based arithmetic checks, convert numbers to strings and combine `len`, `sum`, `zip`, and per-digit integer conversion.
- For filtering, case toggling, splitting, sublist sorting, and interleaving, use comprehensions over the natural iterable structure, often with `zip(*iterables)`.
- For subsequence optimization, use DP arrays storing “best length ending here” or paired forward/backward tables for increasing/decreasing phases.
- For direct helper-style tasks, prefer matching standard library tools: `sorted`, `max/min`, `math.factorial`, `heapq.nlargest`, `sys.getsizeof`, and `re.findall/finditer`.
