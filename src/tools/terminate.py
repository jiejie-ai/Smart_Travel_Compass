from langchain.tools import tool


@tool(description="""Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work.""")
def do_terminate() -> str:
    return "任务结束"
