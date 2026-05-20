"""Claude Code lifecycle hooks — `~/.claude/hooks/claude-code/...` install target.

This package is the source of the four hook scripts Claude Code invokes
over the lifecycle of a session (SessionStart / UserPromptSubmit /
PostToolUse / SessionEnd). The scripts themselves live as siblings of
:mod:`lib`; they import from :mod:`lib` for the sidecar HTTP client,
the stdin-parser models, the rotating-file logger, and the scope
detector wrapper.

See :mod:`hooks.claude_code.lib` for the shared infrastructure.
"""
