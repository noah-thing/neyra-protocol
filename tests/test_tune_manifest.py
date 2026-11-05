import hashlib
import json

from neyra.tune import manifest


def test_file_record_is_the_digest_of_the_bytes(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"one row\n")
    record = manifest.file_record(path)
    assert record["bytes"] == 8
    assert record["sha256"] == "sha256:" + hashlib.sha256(b"one row\n").hexdigest()


def test_model_record_reads_a_folder_and_its_revision(tmp_path):
    folder = tmp_path / "models--a--b" / "snapshots" / "abc123"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    record = manifest.model_record(str(folder))
    assert record["revision"] == "abc123" and record["config.json"].startswith("sha256:")
    assert record["chat_template"] == manifest.digest(b"{{ messages }}")


def test_a_separate_template_file_wins(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("jinja")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "json"}))
    assert manifest.chat_template(tmp_path) == "jinja"


def test_a_name_that_is_nowhere_keeps_only_the_name():
    assert manifest.model_record("nobody/no-such-model-here") == {"name": "nobody/no-such-model-here"}


def test_environment_names_the_tools():
    env = manifest.environment()
    assert env["neyra"] and env["python"] and isinstance(env["packages"], dict)
