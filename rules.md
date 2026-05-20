<!--
rules.md — evasion rules injected verbatim into Gemma's system prompt.

  * HTML comments like this one are STRIPPED before the prompt is sent to the
    model. Use them for notes-to-self; they never reach Gemma.
  * Everything else in this file is appended to the system prompt, after the
    fixed task framing defined in evade.py.
  * Baseline run: no rules. The agent rewrites this file each iteration.
-->
