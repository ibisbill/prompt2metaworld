"""
Shared Azure OpenAI API call helper.

Both the CoT baseline and the LLM-MPC controller use this so retry
and rate-limit handling lives in one place.
"""

from __future__ import annotations

import re
import time
import requests


def post_with_retry(
    endpoint: str,
    headers: dict,
    payload: dict,
    max_wait: int = 40,
    timeout: int = 45,
) -> dict:
    """POST to an Azure OpenAI endpoint, retrying on rate-limit errors.

    Args:
        endpoint:  Full Azure OpenAI chat-completions URL.
        headers:   Request headers (Content-Type + api-key).
        payload:   JSON payload dict (messages, max_tokens, temperature, …).
        max_wait:  Maximum seconds to wait on a rate-limit response.
        timeout:   HTTP request timeout in seconds.

    Returns:
        Parsed JSON response dict. Callers extract
        ``result["choices"][0]["message"]["content"]`` as needed.
    """
    while True:
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            result = resp.json()
        except Exception:
            time.sleep(2)
            continue

        if "error" in result:
            msg = result["error"]["message"]
            match = re.search(r"retry after (\d+) second", msg, re.I)
            wait = int(match.group(1)) if match else 5
            time.sleep(min(wait, max_wait) + 1.0)
        else:
            return result
