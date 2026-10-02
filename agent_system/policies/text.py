"""Text policy adapter, independent of expanded-action implementations."""


class TextStepPolicy:
    async def prepare(self, loop, session, kwargs):
        pass

    def sampling_params(self, params, step_index, max_steps):
        return dict(params)

    def trace(self, generated):
        return {}, None

    def selected_action_names(self, payload, environment):
        return None

    @property
    def extra_fields(self):
        return {"action_interface": "text"}
