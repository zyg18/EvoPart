"""Custom claude_code provider: stdin prompt delivery, a size cap, and capture.

Registers a ClaudeCodeLLM subclass under the same provider name, changing three
things about how each LLM call is made - none of them touching openevolve/:

1. The prompt is piped to the CLI's stdin instead of being passed as an argv
   element. The stock backend appends it to the command line, and Linux caps a
   single argv string at 128 KiB (MAX_ARG_STRLEN): a prompt carrying a ~100 KB
   program plus its history crosses that and dies with E2BIG before the model
   ever sees it. A pipe has no such limit and delivers the bytes unaltered.
2. A 200 KB pre-flight cap on the prompt. Checked in Python before the process
   is spawned - no tokens are spent on an over-budget prompt - and failing
   loudly beats truncating, because a truncated prompt would break the
   SEARCH/REPLACE contract silently. The cap is a cost gate, not a model limit
   (200 KB is ~50K tokens against a 1M-token context).
3. Every request/response pair is written to `llm_calls/` before the framework
   judges it, so discarded iterations remain diagnosable (show_llm.py replays
   the framework's own parsing over these files).

The patch reaches the evaluation workers because the process pool inherits the
default start method (fork on Linux), so children get the patched registry as
part of the forked address space. Installing the hook is one call from
start_evolution.py.

One file per call rather than a shared JSONL: three workers append concurrently
and a 60 KB line is far past the size POSIX guarantees to be atomic, so a
single log file would interleave.
"""

import asyncio
import datetime
import itertools
import json
import os
import pathlib
import subprocess

_counter = itertools.count()

# Pre-flight ceiling on the user prompt, in UTF-8 bytes. Sized so a long run
# is not cut short by prompt growth: prompts peak around 140 KB and
# grow with code size and history, so 250 KB leaves roughly 75% headroom.
PROMPT_CAP_BYTES = 250_000


def install(log_dir: pathlib.Path) -> None:
    """Replace the `claude_code` provider with the stdin + capture subclass."""
    from openevolve.llm import ensemble
    from openevolve.llm.claude_code import ClaudeCodeLLM

    log_dir.mkdir(parents=True, exist_ok=True)

    class CapturingClaudeCodeLLM(ClaudeCodeLLM):
        async def generate_with_context(self, system_message, messages, **kwargs):
            user_content = "\n\n".join(
                m.get("content", "") for m in messages if m.get("role") == "user"
            )
            record = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "pid": os.getpid(),
                "model": self.model,
                "system": system_message or "",
                "user": user_content,
            }
            try:
                response = await self._generate_via_stdin(
                    system_message or "", user_content, **kwargs
                )
                record["response"] = response
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                record["error"] = f"{type(exc).__name__}: {exc}"
                _write(log_dir, record)
                raise
            _write(log_dir, record)
            return response

        async def _generate_via_stdin(self, system_message, user_content, **kwargs):
            nbytes = len(user_content.encode("utf-8"))
            if nbytes > PROMPT_CAP_BYTES:
                # Deliberately outside the retry loop: a too-big prompt does not
                # shrink on retry, and the model must never be billed for it.
                raise RuntimeError(
                    f"prompt is {nbytes} bytes, over the {PROMPT_CAP_BYTES}-byte cap; "
                    f"call refused before reaching the model"
                )

            cmd = [
                "claude", "-p",
                "--model", self.model,
                "--no-session-persistence",
                "--output-format", "text",
                # No tools: the model must answer from the prompt alone. Without
                # this the CLI lets it read arbitrary files on disk (evaluator,
                # other runs' programs, reference sources), which is slow, costly
                # and contaminates the experiment. Anything that would ask for
                # permission is denied rather than left waiting.
                "--tools", "",
                "--permission-prompts", "none",
            ]
            if system_message:
                cmd.extend(["--system-prompt", system_message])
            budget = kwargs.get("max_budget_usd", self.max_budget_usd)
            cmd.extend(["--max-budget-usd", str(budget)])
            # No positional prompt argument: with it absent, `claude -p` reads
            # the prompt from stdin, which argv size limits cannot touch.

            timeout = kwargs.get("timeout", self.timeout)
            retries = kwargs.get("retries", self.retries)
            retry_delay = kwargs.get("retry_delay", self.retry_delay)

            loop = asyncio.get_event_loop()
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: self._run_cli_stdin(cmd, user_content, timeout)
                        ),
                        timeout=timeout + 30,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < retries:
                        await asyncio.sleep(retry_delay)
            raise last_exc

        def _run_cli_stdin(self, cmd, user_content, timeout):
            try:
                result = subprocess.run(
                    cmd,
                    input=user_content,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.cwd,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"claude CLI exceeded {timeout}s")

            output = (result.stdout or "").strip()
            if not output:
                raise RuntimeError(
                    f"empty response from claude CLI (exit {result.returncode}). "
                    f"stderr: {(result.stderr or '')[:500]}"
                )
            return output

    ensemble._PROVIDER_REGISTRY["claude_code"] = lambda cfg: CapturingClaudeCodeLLM(cfg)


def _write(log_dir: pathlib.Path, record: dict) -> None:
    name = f"{record['ts'].replace(':', '')}-{record['pid']}-{next(_counter)}.json"
    (log_dir / name).write_text(json.dumps(record))
