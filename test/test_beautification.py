"""
Test beautification function to verify it handles both single and double quotes
"""

import re

def beautify_response(response: str) -> str:
    """
    Clean and beautify model responses by removing metadata and formatting tags.
    """
    if not isinstance(response, str):
        response = str(response)

    # Remove content=' or content=" wrapper and metadata (from LangChain/Ollama responses)
    # Try with single quotes first
    if "content='" in response:
        match = re.search(r"content='(.*?)' additional_kwargs=", response, re.DOTALL)
        if match:
            response = match.group(1)
    # Try with double quotes
    elif 'content="' in response:
        match = re.search(r'content="(.*?)" additional_kwargs=', response, re.DOTALL)
        if match:
            response = match.group(1)
        else:
            # Handle case where content=" is at the start without closing metadata
            match = re.search(r'content="(.*?)"\s*$', response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                # Just remove the content=" prefix if no pattern matches
                response = re.sub(r'^content="', '', response)
                response = re.sub(r'"\s*$', '', response)

    # Remove <think> tags and their content (from reasoning models)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

    # Remove metadata patterns
    response = re.sub(r"additional_kwargs=\{.*?\}", '', response, flags=re.DOTALL)
    response = re.sub(r"response_metadata=\{.*?\}", '', response, flags=re.DOTALL)
    response = re.sub(r"usage_metadata=\{.*?\}", '', response, flags=re.DOTALL)
    response = re.sub(r"id='run-[^']*'", '', response)

    # Clean up excessive separators
    response = re.sub(r'-{3,}', '\n\n---\n\n', response)
    response = re.sub(r'={3,}', '\n\n---\n\n', response)

    # Clean up excessive newlines
    response = re.sub(r'\n{4,}', '\n\n', response)

    # Remove escape characters
    response = response.replace('\\n', '\n')
    response = response.replace('\\t', '  ')

    # Fix common formatting issues
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)

    # Improve list formatting
    response = re.sub(r'\n-\s+', '\n- ', response)
    response = re.sub(r'\n\*\s+', '\n* ', response)

    # Strip leading/trailing whitespace
    response = response.strip()

    # If response contains Python object syntax, extract content
    if "Message(" in response or "role=" in response:
        lines = []
        for line in response.split('\n'):
            if any(skip in line for skip in ['additional_kwargs', 'response_metadata', 'usage_metadata', 'Message(', 'role=']):
                continue
            lines.append(line)
        response = '\n'.join(lines).strip()

    return response


# Test cases
print("=" * 80)
print("TEST 1: Response with double quotes (content=\")")
print("=" * 80)
test1 = '''content="

Figure 2 illustrates the effectiveness of the proposed online learning method in selecting significant medical images for training. The figure compares the training and validation loss curves obtained using the online approach with those from traditional batch learning methods on datasets such as the NIH Chest X-ray Dataset and the HAM-51 dataset. This comparison demonstrates that the online method achieves better convergence and maintains stability across different data sizes, effectively mitigating catastrophic forgetting while improving data efficiency."
'''
result1 = beautify_response(test1)
print("RESULT:")
print(result1)
print()

print("=" * 80)
print("TEST 2: Response with single quotes (content=')")
print("=" * 80)
test2 = "content='This is a test response with single quotes' additional_kwargs={}"
result2 = beautify_response(test2)
print("RESULT:")
print(result2)
print()

print("=" * 80)
print("TEST 3: Clean response (no metadata)")
print("=" * 80)
test3 = "This is a clean response without any metadata."
result3 = beautify_response(test3)
print("RESULT:")
print(result3)
print()

print("=" * 80)
print("TEST 4: Response with <think> tags")
print("=" * 80)
test4 = '''content="<think>This is internal reasoning</think>

This is the actual response that should be shown."
'''
result4 = beautify_response(test4)
print("RESULT:")
print(result4)
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Test 1 (double quotes): {'PASS' if 'content=' not in result1 else 'FAIL'}")
print(f"✓ Test 2 (single quotes): {'PASS' if 'content=' not in result2 else 'FAIL'}")
print(f"✓ Test 3 (clean): {'PASS' if result3 == test3 else 'FAIL'}")
print(f"✓ Test 4 (<think>): {'PASS' if '<think>' not in result4 else 'FAIL'}")
