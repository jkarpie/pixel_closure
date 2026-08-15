"""Contract-driven electroweak source audit for the NNPDF full closure."""

from pixel.data.closure_audit import audit_electroweak_sources, write_audit
from . import config as cfg


def build():
    return audit_electroweak_sources(cfg.FITPACK_ROOT)


def main():
    path = write_audit(build(), cfg.DATA_ROOT / "ew_audit.json")
    print(f"electroweak audit -> {path}")


if __name__ == "__main__":
    main()
