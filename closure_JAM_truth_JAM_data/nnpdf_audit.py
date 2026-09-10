"""Contract-driven NNPDF commondata audit for the JAM full closure."""

from pixel.data.closure_audit import audit_nnpdf_sources, write_audit
from . import config as cfg


def build():
    return audit_nnpdf_sources(
        cfg.NNPDF_COMMONDATA,
        e906_acceptance=cfg.FITPACK_DY / "SQ_Acceptance.xlsx",
    )


def main():
    path = write_audit(build(), cfg.DATA_ROOT / "nnpdf_audit.json")
    print(f"NNPDF commondata audit -> {path}")


if __name__ == "__main__":
    main()
