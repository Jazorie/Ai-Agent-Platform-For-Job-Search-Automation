#retry logic with exponential backoff for tool calls.
#wraps up any tool function with automatic retry and failure recording.

import asyncio
import time
from functools import wraps

#How many times to retry a failed tool call before giving up
MAX_RETRIES = 3 

#Base wait time in seconds, doubles each retry

BASE_DELAY = 1.0

async def run_with_retry(fn, kwargs: dict, tool_name: str) -> dict:
    #runs a tool function with exponential backoff retry

    #returns a result dict, either the tools output or its success
    #or an error dict with details on final failure
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            # Run the sync tool function in a thread
            # so it doesn't block the async event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: fn(**kwargs)
            )
            
            #if we get here, it worked
            if attempt > 0:
                print(f"[retry] {tool_name} succeeded on attempt {attempt + 1}")

            return {"success": True, "data": result, "attempts": attempt +1}
        except Exception as e:
            last_error = e
            wait = BASE_DELAY * (2 ** attempt) #1s, 2s, 4s
            print(f"[retry] {tool_name} failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

            if attempt < MAX_RETRIES -1:
                print(f"[retry] waiting {wait}s before retry...")
                await asyncio.sleep(wait)
    #All retries exhausted
    print(f"[retry] {tool_name} failed all {MAX_RETRIES} attempts, recording failure")
    return {
        "success": False,
        "error": str(last_error),
        "attempts": MAX_RETRIES,
        "tool_name": tool_name,
    }
def is_failure(result: dict) -> bool:
    #check if a tool result represents a failure
    return isinstance(result, dict) and result.get("success") is False
def unwrap(result: dict):
    #extract the actual data from a successful result
    #if it fails, return the error dict so the agent can handle it
    if result.get("success"):
        return result["data"]
    return result