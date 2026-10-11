"""Check an execution checkpoint immediately before each native input call.

The caller owns policy/cancellation; this bridge contains no planning logic.
"""


class GuardedOSAdapter:
    MUTATIONS = frozenset({
        "launch_app", "focus_window", "focus_text_input", "inject_unicode_text",
        "click", "double_click", "move_mouse", "drag_and_drop", "key_combination",
        "press_key", "scroll", "move_text_caret_to_end", "open_file", "media_control",
    })

    def __init__(self, adapter, checkpoint):
        self.adapter = adapter
        self.checkpoint = checkpoint
        self.mutations = 0

    def __getattr__(self, name):
        method = getattr(self.adapter, name)
        if name not in self.MUTATIONS:
            return method

        def guarded(*args, **kwargs):
            self.checkpoint(name)
            self.mutations += 1
            return method(*args, **kwargs)

        return guarded
