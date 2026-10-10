"""Quick start guides API"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["guides"])

_GUIDES_DIR = Path(__file__).resolve().parent.parent.parent / "guides"

# Guide metadata (order = sidebar order)
GUIDE_LIST = [
    {"name": "overview", "title": "Overview"},
    {"name": "develop-eval-agent", "title": "Build EvalAgent"},
    {"name": "develop-target-agent", "title": "Build TargetAgent"},
    {"name": "generate-benchmark", "title": "Gen Benchmark"},
    {"name": "run-benchmark", "title": "Run Benchmark"},
]

_VALID_NAMES = {g["name"] for g in GUIDE_LIST}


@router.get("/guides")
async def list_guides() -> list[dict]:
    return GUIDE_LIST


@router.get("/guides/{guide_name}")
async def get_guide(guide_name: str) -> dict:
    if guide_name not in _VALID_NAMES:
        raise HTTPException(status_code=404, detail=f"Guide not found: {guide_name}")
    md_path = _GUIDES_DIR / f"{guide_name}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Guide file not found: {guide_name}")
    content = md_path.read_text(encoding="utf-8")
    title = next((g["title"] for g in GUIDE_LIST if g["name"] == guide_name), guide_name)
    return {"name": guide_name, "title": title, "content": content}
