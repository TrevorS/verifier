"""
prompt management module for handling instruction formatting and examples.
"""


def format_example(input_text: str, output: str) -> str:
    """format a single example for few-shot learning."""
    return f"""input: {input_text}
output: {output}"""


def format_examples(examples: list[dict[str, str]]) -> str:
    """format multiple examples for few-shot learning."""
    formatted = []
    for example in examples:
        formatted.append(format_example(example["input"], example["output"]))
    return "\n\n".join(formatted)


def create_prompt(
    input_text: str,
    examples: list[dict[str, str]] | None = None,
    instruction_prefix: str = """convert this verbal monetary expression into a pipe-delimited amount in the format: dollars|cents""",
) -> str:
    """
    create a complete prompt with optional examples.

    args:
        input_text: the input text to process
        examples: optional list of example dictionaries with 'input' and 'output' keys
        instruction_prefix: the instruction to prepend

    returns:
        str: the complete prompt with instruction and optional examples
    """
    parts = []

    # add examples at the top if provided
    if examples:
        parts.extend(["examples:", format_examples(examples), ""])

    # add instruction and input at the bottom
    parts.extend([instruction_prefix, f"input: {input_text}"])
    return "\n".join(parts)
