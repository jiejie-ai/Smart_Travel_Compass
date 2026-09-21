from src.tools.terminate import do_terminate
from src.tools.file_operation import write_file, read_file


def test_terminate():
    result = do_terminate.invoke({})
    assert result == "任务结束"


def test_write_read_file():
    result = write_file.invoke({"file_name": "test_hello.txt", "content": "hello world"})
    assert "successfully" in result
    content = read_file.invoke({"file_name": "test_hello.txt"})
    assert content == "hello world"
