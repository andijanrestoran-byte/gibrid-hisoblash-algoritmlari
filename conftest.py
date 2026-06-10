"""Pytest konfiguratsiyasi: loyiha ildizini import yo'liga qo'shadi."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
