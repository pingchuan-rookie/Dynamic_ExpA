# Official benchmark inputs and local prompts

This document shows each environment's official prompt or native input and its corresponding local prompt.

## ALFWorld

### Official prompt / native input

The native ALFWorld environment provides the task description, current observation and legal actions; the agent submits an action string.

```text
observation: current environment observation, including the goal after Your task is to:
admissible_commands: list of currently executable action strings
agent action: one action string from that list
```

AgentGym's ALFWorld prompt:

```text
Interact with a household to solve a task.
Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal.
At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given a list of actions which you can choose one to perform in this turn.
You should choose from two actions: "THOUGHT" or "ACTION".
If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn.
```

This adapter uses `Thought:` / `Action:` formatting; the local GiGPO-style prompt uses `<think>` / `<action>` tags.

### Local prompt (only the opening sentence changes the official GiGPO template)

The no-history template follows the official structure without a separate task-goal field: the first step relies on the task description in the initial observation. If history is disabled and later observations omit the goal, the input may no longer contain it. This preserves upstream behavior without an additional prompt.

With history:

```text
You are an expert agent operating in an interactive household environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation.
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

Without history:

```text
You are an expert agent operating in an interactive household environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation.
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

## WebShop

### Official prompt / native input

The native WebShop environment provides the shopping instruction, page observation and available actions; the agent submits search or click actions.

```text
observation: current page observation, including the shopping goal after Instruction:
available_actions: whether search is available and the list of currently clickable buttons
agent action: search[keywords] or click[value]
```

AgentGym's WebShop prompt:

```text
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought:
I think ...

Action:
click[something]
```

This adapter uses `Thought:` / `Action:` formatting; the local GiGPO-style prompt uses `<think>` / `<action>` tags.

### Local prompt (only the opening sentence changes the official GiGPO template)

With history:

```text
You are an expert autonomous agent operating in an online shopping environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal.
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

Without history:

```text
You are an expert autonomous agent operating in an online shopping environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal.
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

## DIVE

### Official prompt / native input

DIVE's official task solver sends the question directly as a user message. `{available_actions}` contains full tool schemas (names, descriptions and parameter definitions), also supplied through `tools` to the model's native tool template. There is no additional shared system prompt.

```text
user: {query}
tools: {tool schemas}
```

### Local prompt

The local prompt retains GiGPO-style role, task, observation, bounded history and reasoning instructions. The task comes from the public `query`; observations and history come from tool interaction. These wrappers are not additional official DIVE data fields.

With history:

```text
You are an expert agent solving a task using the available tools.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available tools are: {available_actions}
Use these tool schemas to construct calls; the interface specifies the native tool-call syntax.

Reason about the next step within <think> </think> tags.
Then, directly after </think>, call the appropriate tools using the native tool-call format and their declared arguments, and wait for their results.
When ready, write your final answer directly after </think> instead of tool calls, in the format requested by the query.
A response without tool calls ends the task; do not output reasoning alone or text outside the <think> block other than tool calls or the final answer.
```

Without history:

```text
You are an expert agent solving a task using the available tools.
Your task is to: {task_description}
Your current observation is: {current_observation}
Your available tools are: {available_actions}
Use these tool schemas to construct calls; the interface specifies the native tool-call syntax.

Reason about the next step within <think> </think> tags.
Then, directly after </think>, call the appropriate tools using the native tool-call format and their declared arguments, and wait for their results.
When ready, write your final answer directly after </think> instead of tool calls, in the format requested by the query.
A response without tool calls ends the task; do not output reasoning alone or text outside the <think> block other than tool calls or the final answer.
```

`{available_actions}` contains full tool schemas (names, descriptions and parameter definitions), also passed through `tools` to the native tool template. `{current_observation}` is the latest tool feedback, and `{action_history}` is the bounded observation/action history.

## CodeGym

### Official prompt / native input

CodeGym's system message contains function declarations and docstrings for the current environment; the user message contains the following rules and task text.

```text
Please answer the following question step by step according to the requirements below!

1. **Do not** write code to answer the user's question - you may only call the provided functions, and you may call at most **one function per step**.
2. After you call a function, wait for the tool to return the result - do not assume what the result will be.
3. If the tool’s description is unclear, you can try using it first, and then adjust your function call based on the returned result.
4. Function calls should be wrapped with `<|FunctionCallBegin|>....<|FunctionCallEnd|>` and contain a JSON-formatted list.
The list should include **one dictionary**, where each dictionary contains two parameters:

   * `name`: the function name
   * `parameters`: a dictionary of key-value pairs for the arguments

Here’s an example of a function call:
<|FunctionCallBegin|>[{"name":"function_name", "parameters":{"key1":"value1", "key2":"value2"}}]<|FunctionCallEnd|>

**Extra requirements:**

* Do not overthink; think briefly, then decide how to call the function.
* Since you have many chances to call functions, you do not need to plan all steps in advance.
* Do not try to solve the problem without using the tools.

Question:
{gym_task}
```

### Local prompt

Function declarations and descriptions remain in the system message. The following templates are populated with the task, actual observations and bounded history.

With history:

```text
You are an expert agent operating in an interactive function-calling environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available functions are: [{available_actions}].

Reason about the next step within <think> </think> tags.
Then output exactly one function call using its declared arguments:
<|FunctionCallBegin|>[{"name": "<function name>", "parameters": {"<argument name>": "<value>"}}]<|FunctionCallEnd|>
Use {} for no arguments; do not write executable code.
Wait for feedback after each call and follow the task's submission rules.
```

Without history:

```text
You are an expert agent operating in an interactive function-calling environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available functions are: [{available_actions}].

Reason about the next step within <think> </think> tags.
Then output exactly one function call using its declared arguments:
<|FunctionCallBegin|>[{"name": "<function name>", "parameters": {"<argument name>": "<value>"}}]<|FunctionCallEnd|>
Use {} for no arguments; do not write executable code.
Wait for feedback after each call and follow the task's submission rules.
```

## t2bench

### Official prompt / native input

t2bench's official standard agent uses the following system prompt, with tool schemas supplied separately through `tools`.

```text
<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
Always make sure you generate valid JSON only.
</instructions>
<policy>
{domain_policy}
</policy>
```

`domain_policy` contains the public rules for the current domain; the conversation supplies the customer's request.

### Local prompt

System prompt：

```text
{domain_policy}
#Available tools
{JSON serialized official tool schemas}
# Instruction
Follow the policy above and the declared tool schemas.
Ask the customer for missing information or required confirmation; do not invent facts or tool results.
```

With history:

```text
You are an expert agent helping a customer in an interactive service environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: [{available_actions}].

Reason about the next step within <think> </think> tags.
Then output exactly one action as valid JSON enclosed within <action> </action> tags:
<action>
{"name": "<action name>", "arguments": {"<argument name>": "<value>"}}
</action>
To message the customer, use name="respond" and arguments={"content": "<message>"}; this does not end the conversation.
Stop after </action> and wait for feedback; output no text outside the <think> and <action> blocks or Markdown fences.
```

Without history:

```text
You are an expert agent helping a customer in an interactive service environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: [{available_actions}].

Reason about the next step within <think> </think> tags.
Then output exactly one action as valid JSON enclosed within <action> </action> tags:
<action>
{"name": "<action name>", "arguments": {"<argument name>": "<value>"}}
</action>
To message the customer, use name="respond" and arguments={"content": "<message>"}; this does not end the conversation.
Stop after </action> and wait for feedback; output no text outside the <think> and <action> blocks or Markdown fences.
```

`{task_description}` comes from the customer's first public message. `{current_observation}` is the latest customer message or tool feedback, and `{action_history}` is the bounded observation/action history.
The local protocol uses JSON inside `<action>` to call tools or reply to the customer through `respond`.

## SWE-bench Verified

### Official prompt / native input

SWE-bench Verified supplies a software issue, its repository and a base commit; it does not mandate a shared agent prompt or tool interface.

```text
problem_statement: public issue text
repo: repository identifier
base_commit: base commit of the repository to repair
instance_id: task identifier
submission: model_patch generated against that repository
```

Reference fixes, hidden test patches and scoring fields are excluded from the agent prompt.

### Local prompt

System prompt：

```text
You are fixing a software issue in an isolated, offline repository at /testbed.
Inspect the repository, implement the requested fix, and run relevant local tests.
Use the provided bash, search, and editor tools.
Dependencies are already installed; network access is unavailable.
Shell commands start in /testbed and do not persist shell state between calls.
The testbed conda environment is activated when available.
Your final submission is the actual working-tree diff, including new files, not text in your response.
Do not edit Git metadata or rely on repository history.
Hidden evaluation tests and reference solutions are unavailable.
Call finish when ready.
```

With history:

```text
You are an expert agent fixing a software issue in an offline repository.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available tools are: {available_actions}
Their argument schemas and native tool-call format are provided by the interface.
Follow the repository and submission instructions above.

Now it's your turn to take the next step.
You should first reason step-by-step about the reported issue, the current evidence, and what to inspect, change, or test next.
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, make your tool calls directly after </think>, using the native tool-call format and their declared argument schemas.
After making tool calls, stop and wait for their results before continuing. Output no text outside the <think> block and the tool calls.
Use repository contents and local tool results as evidence; do not invent file contents or test outcomes.
When ready, call finish to submit the actual working-tree diff, including new files, for separate evaluation.
If you make multiple tool calls in one response, finish must be the last call.
A plain-text answer or reasoning alone does not submit the patch and is not a valid action.
```

Without history:

```text
You are an expert agent fixing a software issue in an offline repository.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available tools are: {available_actions}
Their argument schemas and native tool-call format are provided by the interface.
Follow the repository and submission instructions above.

Now it's your turn to take the next step.
You should first reason step-by-step about the reported issue, the current evidence, and what to inspect, change, or test next.
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, make your tool calls directly after </think>, using the native tool-call format and their declared argument schemas.
After making tool calls, stop and wait for their results before continuing. Output no text outside the <think> block and the tool calls.
Use repository contents and local tool results as evidence; do not invent file contents or test outcomes.
When ready, call finish to submit the actual working-tree diff, including new files, for separate evaluation.
If you make multiple tool calls in one response, finish must be the last call.
A plain-text answer or reasoning alone does not submit the patch and is not a valid action.
```

`{task_description}` contains the public issue, repository and task identifier. `{current_observation}` is the latest tool feedback, and `{action_history}` is the bounded observation/action history.
Tool schemas are supplied through `tools`; the baseline and Dyad use the same template.

## GSM8K

### Official prompt / native input

Official GSM8K data supplies a mathematical `question` and an `answer` containing the solution and final value after `####`. It does not define the local calculator action or history protocol.

```text
question: {question}
answer: {reference solution}
#### {reference numeric answer}
```

The model reads only `question`; the reference solution and final value are used only for scoring.

### Local prompt

Local evaluation uses multi-turn calculator interaction: `calculate` returns a computation result, and `answer` submits the value and ends the task.

With history:

```text
You are an expert agent solving a math problem with a calculator.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: calculate and answer.

Reason about the next step within <think> </think> tags, then output exactly one action.
Use <action>calculate {expression}=</action> for arithmetic and wait for the result.
Use <action>answer {number}</action> to submit the final numeric answer and end the task.
```

Without history:

```text
You are an expert agent solving a math problem with a calculator.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: calculate and answer.

Reason about the next step within <think> </think> tags, then output exactly one action.
Use <action>calculate {expression}=</action> for arithmetic and wait for the result.
Use <action>answer {number}</action> to submit the final numeric answer and end the task.
```

`{task_description}` is the public question, `{current_observation}` is the latest calculator feedback, and `{action_history}` is the bounded observation/action history.
Actions and reasoning use the shared lowercase `<action>` and `<think>` tags, respectively.
