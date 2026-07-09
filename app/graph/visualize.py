"""Save the LangGraph workflow visualization as PNG and Mermaid."""

from pathlib import Path

from langchain_core.runnables.graph import CurveStyle, NodeStyles

from app.graph.builder import build_graph


DEFAULT_OUTPUT_DIR = Path("app")
DEFAULT_PNG_PATH = DEFAULT_OUTPUT_DIR / "agent_graph.png"
DEFAULT_MERMAID_PATH = DEFAULT_OUTPUT_DIR / "agent_graph.mmd"

GRAPH_THEME = {
    "config": {
        "theme": "base",
        "themeVariables": {
            "primaryColor": "#EEF2FF",
            "primaryTextColor": "#111827",
            "primaryBorderColor": "#6366F1",
            "lineColor": "#64748B",
            "secondaryColor": "#ECFDF5",
            "tertiaryColor": "#FFF7ED",
            "fontFamily": "Inter, Arial, sans-serif",
        },
    }
}

NODE_COLORS = NodeStyles(
    default="fill:#EEF2FF,stroke:#6366F1,color:#111827,line-height:1.25",
    first="fill:#DCFCE7,stroke:#16A34A,color:#052E16",
    last="fill:#FFE4E6,stroke:#E11D48,color:#4C0519",
)


def save_graph_visualization(
    png_path: str | Path = DEFAULT_PNG_PATH,
    mermaid_path: str | Path = DEFAULT_MERMAID_PATH,
) -> None:
    """Render the compiled graph to PNG and Mermaid source files."""

    graph = build_graph()
    drawable_graph = graph.get_graph()

    mermaid_text = drawable_graph.draw_mermaid(
        curve_style=CurveStyle.BASIS,
        node_colors=NODE_COLORS,
        wrap_label_n_words=4,
        frontmatter_config=GRAPH_THEME,
    )

    png_bytes = drawable_graph.draw_mermaid_png(
        curve_style=CurveStyle.BASIS,
        node_colors=NODE_COLORS,
        wrap_label_n_words=4,
        background_color="#FFFFFF",
        padding=24,
        frontmatter_config=GRAPH_THEME,
    )

    png_path = Path(png_path)
    mermaid_path = Path(mermaid_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_path.parent.mkdir(parents=True, exist_ok=True)

    png_path.write_bytes(png_bytes)
    mermaid_path.write_text(mermaid_text, encoding="utf-8")


if __name__ == "__main__":
    save_graph_visualization()
