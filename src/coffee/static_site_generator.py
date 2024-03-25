import math
import pathlib
import shutil
from collections import defaultdict
from functools import singledispatchmethod
from typing import Mapping

import casefy
import tomli
from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup

from coffee.benchmark import BenchmarkDefinition, BenchmarkScore, HazardDefinition
from coffee.newhelm_runner import NewhelmSut
from coffee.utilities import group_by_key

from newhelm.base_test import BaseTest


def display_stars(score, size) -> Markup:
    empty_stars = 5 - score
    stars_html = f"""
    <span class="star-span-{size}"><svg xmlns="http://www.w3.org/2000/svg" fill="#596C97"
    class="bi bi-star-fill"
    viewBox="0 0 16 16">
    <path d="M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.282.95l-3.522 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z"/>
    </svg></span>
    """
    empty_stars_html = f"""
    <span class="star-span-{size}"><svg xmlns="http://www.w3.org/2000/svg" fill="#C8CFDD"
    class="bi bi-star-fill"
    viewBox="0 0 16 16">
    <path d="M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.282.95l-3.522 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z"/>
    </svg></span>
    """
    final_html = (stars_html * score) + (empty_stars_html * empty_stars)
    return Markup(final_html)


class StaticSiteGenerator:
    def __init__(self, view_embed: bool = False) -> None:
        """Initialize the StaticSiteGenerator class for local file or website partial

        Args:
            view_embed (bool): Whether to generate local file or embedded view. Defaults to False.
        """
        self.view_embed = view_embed
        self.env = Environment(loader=PackageLoader("coffee"), autoescape=select_autoescape())
        self.env.filters["display_stars"] = display_stars
        self.env.globals["root_path"] = self.root_path
        self.env.globals["benchmarks_path"] = self.benchmarks_path
        self.env.globals["benchmark_path"] = self.benchmark_path
        self.env.globals["test_report_path"] = self.test_report_path
        self.env.globals["content"] = self.content
        self._content = self._load_content()

    def _load_content(self):
        content = {}
        for file in (pathlib.Path(__file__).parent / "templates" / "content").rglob("*.toml"):
            with open(file, "rb") as f:
                data = tomli.load(f)
                duplicate_keys = set(content.keys()) & set(data.keys())
                if duplicate_keys:
                    raise Exception(f"Duplicate tables found in content files: {duplicate_keys}")
                content.update(data)
        return content

    @singledispatchmethod
    def content(self, item, key: str):
        pass

    @content.register
    def content_benchmark(self, item: BenchmarkDefinition, key: str):
        return self._content[item.path_name()][key]

    @content.register
    def content_hazard(self, item: HazardDefinition, key: str):
        return self._content[casefy.snakecase(item.__class__.__name__.replace("Definition", ""))][key]

    @content.register
    def content_string(self, item: str, key: str):
        return self._content[item][key]

    @content.register
    def content_sut(self, item: NewhelmSut, key: str):
        return self._content[item.name][key]

    @content.register
    def content_test(self, item: BaseTest, key: str):
        # in this case, we want to default to some sensible fallbacks
        try:
            value = self._content[item.uid][key]
        except KeyError:
            defaults = defaultdict(str)
            defaults["display_name"] = item.uid
            value = defaults[key]
        return value

    def _template_dir(self):
        current_path = pathlib.Path(__file__)
        while not current_path.name == "coffee":
            current_path = current_path.parent
        return current_path / "templates"

    def _static_dir(self):
        return self._template_dir() / "static"

    def _copy_static_dir(self, output_dir):
        shutil.copytree(self._static_dir(), output_dir / "static", dirs_exist_ok=True)

    def generate(
        self,
        benchmark_scores: list[BenchmarkScore],
        output_dir: pathlib.Path,
    ) -> None:
        self._copy_static_dir(output_dir)
        self._generate_index_page(output_dir)
        self._generate_benchmarks_page(benchmark_scores, output_dir)
        self._generate_benchmark_pages(benchmark_scores, output_dir)
        self._generate_test_report_pages(benchmark_scores, output_dir)

    def _render_template(self, template_name, **kwargs):
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    def _write_file(self, output: pathlib.Path, template_name: str, **kwargs) -> None:
        with open(pathlib.Path(output), "w+") as f:
            f.write(self._render_template(template_name, **kwargs))

    def _generate_index_page(self, output_dir: pathlib.Path) -> None:
        self._write_file(
            output=output_dir / "index.html",
            template_name="index.html",
            page_type="index",
            view_embed=self.view_embed,
        )

    def _grouped_benchmark_scores(self, benchmark_scores: list[BenchmarkScore]) -> Mapping:
        return group_by_key(benchmark_scores, lambda x: x.benchmark_definition)

    def _generate_benchmarks_page(self, benchmark_scores: list[BenchmarkScore], output_dir: pathlib.Path) -> None:
        self._write_file(
            output=output_dir / "benchmarks.html",
            template_name="benchmarks.html",
            grouped_benchmark_scores=self._grouped_benchmark_scores(benchmark_scores),
            show_benchmark_header=True,
            page_type="benchmarks",
            view_embed=self.view_embed,
        )

    def _generate_benchmark_pages(self, benchmark_scores: list[BenchmarkScore], output_dir: pathlib.Path) -> None:
        for benchmark_definition, benchmark_scores in self._grouped_benchmark_scores(benchmark_scores).items():
            for benchmark_score in benchmark_scores:
                self._write_file(
                    output=output_dir / f"{benchmark_score.benchmark_definition.path_name()}.html",
                    template_name="benchmark.html",
                    benchmark_definition=benchmark_definition,
                    grouped_benchmark_scores=self._grouped_benchmark_scores(benchmark_scores),
                    page_type="benchmark",
                    view_embed=self.view_embed,
                )

    def _generate_test_report_pages(self, benchmark_scores: list[BenchmarkScore], output_dir: pathlib.Path) -> None:
        for benchmark_score in benchmark_scores:
            self._write_file(
                output=output_dir
                / f"{benchmark_score.sut.name}_{benchmark_score.benchmark_definition.path_name()}_report.html",
                template_name="test_report.html",
                benchmark_score=benchmark_score,
                page_type="test_report",
                view_embed=self.view_embed,
            )

    def root_path(self) -> str:
        return "#" if self.view_embed else "index.html"

    def benchmarks_path(self, page_type: str) -> str:
        if page_type == "benchmarks" and self.view_embed:
            return "../benchmarks"
        return "../../benchmarks" if self.view_embed else "benchmarks.html"

    def benchmark_path(self, benchmark_path_name, page_type: str) -> str:
        if page_type == "benchmarks" and self.view_embed:
            return f"{benchmark_path_name}"
        return f"../{benchmark_path_name}" if self.view_embed else f"{benchmark_path_name}.html"

    def test_report_path(self, sut_path_name: str, benchmark_path_name: str, page_type: str) -> str:
        if page_type == "benchmarks" and self.view_embed:
            return f"{sut_path_name}_{benchmark_path_name}_report"
        return (
            f"../{sut_path_name}_{benchmark_path_name}_report"
            if self.view_embed
            else f"{sut_path_name}_{benchmark_path_name}_report.html"
        )
