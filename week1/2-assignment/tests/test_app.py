from app import load_report, generate_wordcloud


def test_load_report(tmp_path):
    report = tmp_path / "sample.txt"
    report.write_text(
        "AI assists software engineering.",
        encoding="utf-8"
    )

    text = load_report(report)

    assert text == "AI assists software engineering."


def test_generate_wordcloud_creates_output_file(tmp_path):
    output_file = tmp_path / "wordcloud.png"

    generate_wordcloud("AI assists software engineering.", str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0